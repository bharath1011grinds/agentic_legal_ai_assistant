import os
import json
from contextlib import asynccontextmanager  # modern FastAPI lifespan pattern
from dotenv import load_dotenv
 
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from agents.agent_state import initial_state, intake_initial_state
from agents.synthesizer import stream_answer
from pipeline_agentic import build_graph_no_synth, _LANGFUSE_ENABLED
from intake_graph import build_intake_graph, build_decomposition_graph
from intake_retrieval_graph import build_intake_retrieval_graph
from agents.intake_synthesizer import stream_intake_answer


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# Import observed stream_answer if Langfuse is enabled
if _LANGFUSE_ENABLED:
    from observability import (observed_synthesize_stream_node as stream_answer,
                               observed_stream_intake_answer  as stream_intake_answer,
                               shutdown as langfuse_shutdown, flush as langfuse_flush,
                               observed_ask)
else:
    langfuse_flush    = lambda: None
    langfuse_shutdown = lambda: None
 
load_dotenv()

HISTORIES = []
def _format_clarification_for_history(questions: list[str]) -> str:
    """
    Convert the clarifying questions into a readable assistant turn so the
    context_resolver can see what the assistant asked when the user's answer arrives.
 
    Input  : ["What is your situation?|||Police stopped me|I was fired|Landlord issue"]
    Output : "What is your situation? [Police stopped me / I was fired / Landlord issue]"
    """
    parts = []
    for q in questions:
        segments = q.split("|||")
        q_text   = segments[0].strip()
        options  = [o.strip() for o in segments[1].split("|")] if len(segments) > 1 else []
        opts_str = " / ".join(options)
        parts.append(f"{q_text} [{opts_str}]" if opts_str else q_text)
    return "\n".join(parts)

def _get_session(session_id, app) -> dict:

    """
    Returns the full session dict for a session_id.
    Creates it with defaults if it doesn't exist.
    """

    if session_id not in app.state.sessions:
        app.state.sessions[session_id] = {
            "history": [],
            "mode": "quick",
            "case_log": None,
            "intake_turn_count": 0,
            "awaiting_contradiction_resolution": False,
        }
    
    return app.state.sessions[session_id]



@asynccontextmanager
async def lifespan(app: FastAPI):

    #startup
    #build the graph and store it in a state variable graph.
    app.state.graph = build_graph_no_synth()
    app.state.intake_graph            = build_intake_graph()
    app.state.decomposition_graph     = build_decomposition_graph()
    app.state.intake_retrieval_graph  = build_intake_retrieval_graph()

    #Iniitializing the sessions dict, where hist, mode etc.. gets stored according to session ids..
    app.state.sessions = {} #session_id : {history : [], mode: "", ...}
  
    print("[Server] All Graphs Ready!")

    yield

    #shutdown
    langfuse_flush()
    langfuse_shutdown()
    print("[Server] Shutdown complete")


app =FastAPI(title= "KnowYourRights",lifespan=lifespan)


# Allow requests from the UI (served locally or on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#Deffine a pydantic model for the queryrequest
class QueryRequest(BaseModel):
    query: str
    session_id : str = "default"

class ModeRequest(BaseModel):
    session_id : str = "default"
    mode : str  #'quick' | 'intake'



@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


#Health route to check if the app is alive
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": "ready"}


@app.post('/mode')
async def set_mode(payload : ModeRequest, request: Request):

    valid_modes = ["quick", "intake"]

    if payload.mode not in valid_modes:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid mode. Choose from {valid_modes}"}
        )
    session = _get_session(payload.session_id, request.app)
    session["mode"] = payload.mode

    print(f"[Server] Session '{payload.session_id}' mode set to '{payload.mode}'")
    return {"status": "ok", "session_id": payload.session_id, "mode": payload.mode}



@app.post('/query')
async def query(payload: QueryRequest, request: Request):

    #We need an inner fn here, cos FASTAPI MUST either return a JSONResponse or a StreamingResponse
    #When we use streaming we cant return any of the above. So we yield from the inner function and return the 
    #StreamingResponse from the outer fn. 
    async def event_stream():
        #fetch the session_id from the QueryRequest
        session_id = payload.session_id

        #need to pass the request.app also to get_session cos the get_session makes use of the app.state inside
        session = _get_session(session_id, request.app)

        #fetch the history of that session_id from session_id
        history = session['history'][-8:]

        init_state = initial_state(payload.query, history=history)
        #Invoking the graph without the synth node.
        #request object contains a pointer to the state variable "state" ...
        final_state = request.app.state.graph.invoke(init_state)

        # ── Clarification short-circuit ──────────────────────────────────
        if final_state.get("answer") == "__CLARIFICATION__":
            questions = final_state.get("clarification_questions", [])
 
            # Persist to history BEFORE returning to the client.
            # When the user sends their chosen option next turn, the resolver sees:
            #   User:      "what are my rights"
            #   Assistant: "What is your situation? [Police stopped me / I was fired / ...]"
            # ...and resolves "Police stopped me" → "what are my rights when police stop me"
            clarify_text = _format_clarification_for_history(questions)
            print(questions)
            updated_history = history + [
                {"role": "user",      "content": payload.query},
                {"role": "assistant", "content": clarify_text},
            ]
            session["history"] = updated_history 
            yield f"event: clarify\ndata: {json.dumps({'clarifying_questions': questions})}\n\n"
            return
        

        #Handling the Out of Scope condition
        if final_state.get("out_of_scope"):
            answer = final_state.get("answer", "")
            updated_history = history + [{'role': 'user', 'content': payload.query},
                                     {'role': 'assistant', 'content': answer}]
            session["history"] = updated_history 
            yield f"event: scope_guard\ndata: {json.dumps({'answer': answer})}\n\n"
            return
    

        #Handling KB miss scenario
        grader = final_state.get("grader_output")
        if grader and grader.rerouting_decision == "detect":
            answer = final_state.get("answer", "")
            updated_history = history + [{'role': 'user', 'content': payload.query},
                                     {'role': 'assistant', 'content': answer}]
            session["history"] = updated_history 
            #print(history)
            yield f"event: kb_miss\ndata: {answer}\n\n"
            # yield f"event: kb_miss\ndata: {json.dumps({'answer': answer})}\n\n"
            return
        

        #Handling synthesis
        gen = stream_answer(final_state)

        full_answer = ""
        for token in gen:

            #Removing the prefix added in stream_answer() 
            full_answer += token.removeprefix("data:").strip()
            #sending token
            yield token
        
        #add the most recent exchange to the existing history
        updated_history = history + [{'role': 'user', 'content': payload.query},
                                     {'role': 'assistant', 'content': full_answer}]
        
        #update the history for this session_id in the histories...
        session["history"] = updated_history
#----------------------------------------------------------------------------------------------------
       
        #At this point, all the tokens would have been sent to the UI.
        #We gather a bunch of metadata and push them to the UI as a single yield...
        

        # We already streamed the answer above - extract metadata from final_state
        grader_output = final_state.get("grader_output")
        analysis      = final_state.get("analysis")
 
        meta = {
            "citations":  [],   # stream_answer doesn't write back to state, parsed below
            "image_refs": [],
            "stats": {
                "query_type":       analysis.query_type       if analysis      else "N/A",
                #"modality":         analysis.modality         if analysis      else "N/A",
                "retry_count":      final_state.get("retry_count", 0),
                "routing_decision": grader_output.rerouting_decision if grader_output else "N/A",
                "tavily_used":      len(final_state.get("web_results") or []) > 0,
                "chunks_passed":    len(grader_output.passed_chunks)  if grader_output else 0,
            }
        }

        # Extract image_refs directly from chunks (figure captions)
        for doc in final_state.get("chunks", []):
            if doc.metadata.get("content_type") == "figure":
                meta["image_refs"].append({
                    "image_path":  doc.metadata.get("image_path", ""),
                    "caption":     doc.metadata.get("caption_description", ""),
                    "figure_type": doc.metadata.get("figure_type", ""),
                    "title":       doc.metadata.get("title", ""),
                })
 
        # Send final metadata event - UI listens for event: done
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"
    
    #returns a streaming response, with media_type as text/event-stream to prevent tokens from buffering and being pushed ASAP.
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post('/intake/query')
async def intake_query(payload : QueryRequest, request : Request):

    async def event_stream():

        session_id = payload.session_id
        session = _get_session(session_id, request.app)
        history = session["history"][-16:]
        #The below returns a GraphState Object
        init_state = intake_initial_state(raw_query=payload.query,
                                          history=history,
                                          case_log=session['case_log'], 
                                          intake_turn_count=session['intake_turn_count']
                                          )
        
        init_state['awaiting_contradiction_resolution'] = session['awaiting_contradiction_resolution']

        #trigger the intake_graph stored in the app state
        final_state = request.app.state.intake_graph.invoke(init_state)

        #update the session variables after graph run
        session["case_log"]          = final_state.get("case_log")
        session["intake_turn_count"] = final_state.get("intake_turn_count", 0)
        session["awaiting_contradiction_resolution"] = final_state.get(
            "awaiting_contradiction_resolution", False
        )

        #update the history
        answer = final_state.get("answer", "")

        updated_history = history + [
            {"role": "user",      "content": payload.query},
            {"role": "assistant", "content": answer},
        ]

        session["history"] = updated_history

        #if turn_limit hit or all fields filled - refer intake graph fns
        intake_ready = "__INTAKE_READY__" in final_state.get("clarification_questions", [])

        
        #Serialize[convert data to json] case_log for the UI side panel
        case_log      = session["case_log"]
        case_log_dict = case_log.model_dump() if case_log else {}
 
        payload_out = {
            "answer":       answer,
            "intake_ready": intake_ready,
            "case_log":     case_log_dict,
            "turn_count":   session["intake_turn_count"],
        }
 
        yield f"event: intake_turn\ndata: {json.dumps(payload_out)}\n\n"
 
    return StreamingResponse(event_stream(), media_type="text/event-stream")



#Intake mode - Get Legal Advise

@app.post('/intake/advise')
async def intake_advise(payload : QueryRequest, request : Request):

    """
    Triggered when the user clicks "Get Legal Advice".
    1. Runs decomposition graph → populates decomposed_claims
    2. Runs intake retrieval graph → parallel retrieval + grading
    3. Streams the claim-by-claim answer
 
    If case_log was edited after a previous advise call, this endpoint
    re-runs from decomposition automatically (full re-retrieval).
    """

    async def event_stream():
        session    = _get_session(payload.session_id, request.app)
        session_id = payload.session_id
        case_log   = session.get("case_log")


        if case_log is None or not case_log.non_negotiables_filled():
            yield f"event: error\ndata: {json.dumps({'error': 'Case log incomplete. Please complete the intake first.'})}\n\n"
            return
        
        decomp_state = intake_initial_state(
            raw_query = "",
            case_log  = case_log,
        )

        decomp_final = request.app.state.decomposition_graph.invoke(decomp_state)
        decomposed   = decomp_final.get("decomposed_claims")

        #If no decomposed claims are present
        if decomposed is None or not decomposed.claims:
            yield f"event: error\ndata: {json.dumps({'error': 'Could not decompose claims from case log.'})}\n\n"
            return
        
        print(f"[Server] Decomposed {len(decomposed.claims)} claims for session '{session_id}'")


        #Parallel retrieval + grading
        retrieval_state = intake_initial_state(
            raw_query = "",
            case_log  = case_log,
        )
        retrieval_state["decomposed_claims"] = decomposed
 
        retrieval_final = request.app.state.intake_retrieval_graph.invoke(retrieval_state)

        #KB miss case
        grader = retrieval_final.get("grader_output")
        if grader and grader.rerouting_decision == "detect":
            answer = retrieval_final.get("answer", "No relevant legal information found for your situation.")
            session["history"] = session["history"] + [
                {"role": "user",      "content": "[Get Legal Advice]"},
                {"role": "assistant", "content": answer},
            ]
            yield f"event: kb_miss\ndata: {answer}\n\n"
            return

       # ── Step 3: Stream claim-by-claim answer ──────────────────────────────
        gen         = stream_intake_answer(retrieval_final)
        full_answer = ""
        for token in gen:
            full_answer += token.removeprefix("data:").strip()
            yield token
 
        # Persist answer to history
        session["history"] = session["history"] + [
            {"role": "user",      "content": "[Get Legal Advice]"},
            {"role": "assistant", "content": full_answer},
        ]

        #send metadata as an event: done
        claim_results = retrieval_final.get("claim_results", [])
        meta = {
            "citations":    [],
            "claim_results": claim_results,
            "stats": {
                "claims_decomposed": len(decomposed.claims),
                "chunks_passed":     len(grader.passed_chunks) if grader else 0,
                "routing_decision":  grader.rerouting_decision if grader else "N/A",
                "tavily_used":       len(retrieval_final.get("web_results") or []) > 0,
            }
        }
 
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"
 
    return StreamingResponse(event_stream(), media_type="text/event-stream")



#Intake mode - edit case_log
@app.post("/intake/edit")
async def intake_edit(payload: QueryRequest, request: Request):
    """
    Handles natural language edits to the case log after it's been shown
    in the side panel. Reuses the intake graph (edit_detector runs first).
    If the user has already clicked "Get Legal Advice", triggers auto re-retrieval
    after the edit is applied.
    """

    async def event_stream():
        session    = _get_session(payload.session_id, request.app)
        session_id = payload.session_id

        #check if legal advice has already been clicked by the user...
        advice_given = len([
            t for t in session["history"]
            if t.get("content") == "[Get Legal Advice]"
        ]) > 0

        # Run intake graph - edit_detector handles the correction
        init_state = intake_initial_state(
            raw_query         = payload.query,
            history           = session["history"][-8:],
            case_log          = session["case_log"],
            intake_turn_count = session["intake_turn_count"],
        )
        init_state["awaiting_contradiction_resolution"] = session["awaiting_contradiction_resolution"]
 
        final_state = request.app.state.intake_graph.invoke(init_state)

        #Update the session variables
        session["case_log"]          = final_state.get("case_log")
        session["intake_turn_count"] = final_state.get("intake_turn_count", 0)
        session["awaiting_contradiction_resolution"] = final_state.get(
            "awaiting_contradiction_resolution", False
        )

        answer        = final_state.get("answer", "")
        edited_fields = final_state.get("edited_fields", [])
        case_log_dict = session["case_log"].model_dump() if session["case_log"] else {}
 
        session["history"] = session["history"] + [
            {"role": "user",      "content": payload.query},
            {"role": "assistant", "content": answer},
        ]
 
        yield f"event: edit_confirmed\ndata: {json.dumps({'answer': answer, 'edited_fields': edited_fields, 'case_log': case_log_dict})}\n\n"
 

        #Auto re-retrieval if legal advice was already given and an edit is detected in the log
        
        if advice_given and edited_fields:
            print(f"[Server] Log edited post-advice — triggering re-retrieval for '{session_id}'")
 
            case_log = session["case_log"]
            if case_log is None or not case_log.non_negotiables_filled():
                return
            

            decomp_state = intake_initial_state(raw_query="", case_log=case_log)
            decomp_final = request.app.state.decomposition_graph.invoke(decomp_state)
            decomposed   = decomp_final.get("decomposed_claims")
 
            if not decomposed or not decomposed.claims:
                return
            
            retrieval_state = intake_initial_state(raw_query="", case_log=case_log)
            retrieval_state["decomposed_claims"] = decomposed
            retrieval_final = request.app.state.intake_retrieval_graph.invoke(retrieval_state)

            grader = retrieval_final.get("grader_output")
            if grader and grader.rerouting_decision == "detect":
                answer = "No relevant legal information found after the update."
                session["history"].append({"role": "assistant", "content": answer})
                yield f"event: kb_miss\ndata: {answer}\n\n"
                return
 
            gen = stream_intake_answer(retrieval_final)
            full_answer = ""
            for token in gen:
                full_answer += token.removeprefix("data:").strip()
                yield token
 
            session["history"].append({"role": "assistant", "content": full_answer})
 

            #sending metadata as an event : done
            claim_results = retrieval_final.get("claim_results", [])
            meta = {
                "citations":     [],
                "claim_results": claim_results,
                "stats": {
                    "claims_decomposed": len(decomposed.claims),
                    "chunks_passed":     len(grader.passed_chunks) if grader else 0,
                    "routing_decision":  grader.rerouting_decision if grader else "N/A",
                    "tavily_used":       len(retrieval_final.get("web_results") or []) > 0,
                }
            }
            yield f"event: done\ndata: {json.dumps(meta)}\n\n"
 
    return StreamingResponse(event_stream(), media_type="text/event-stream")




@app.post("/reset")
async def reset(request : Request, session_id: str = "default"):

    request.app.state.sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}



