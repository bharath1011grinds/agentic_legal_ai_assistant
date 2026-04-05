import os
import json
from contextlib import asynccontextmanager  # modern FastAPI lifespan pattern
from dotenv import load_dotenv
 
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from agents.agent_state import initial_state
from agents.synthesizer import stream_answer
from pipeline_agentic import build_graph_no_synth, _LANGFUSE_ENABLED
 
# Import observed stream_answer if Langfuse is enabled
if _LANGFUSE_ENABLED:
    from observability import observed_synthesize_stream_node as stream_answer, shutdown as langfuse_shutdown, flush as langfuse_flush
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


@asynccontextmanager
async def lifespan(app: FastAPI):

    #startup
    #build the graph and store it in a state variable graph.
    app.state.graph = build_graph_no_synth()

    #Iniitializing the histories dict, where hist gets stored acc to session ids..
    app.state.histories = {} #session_id : list[dict]
    print("[Server] Pipeline ready")

    yield

    #shutdown
    langfuse_flush()
    langfuse_shutdown()
    print("[Server] Shutdown complete")


app =FastAPI(title= "arXiv Search Assistant",lifespan=lifespan)


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

#Health route to check if the app is alive
@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": "ready"}



@app.post('/query')
async def query(payload: QueryRequest, request: Request):

    #We need an inner fn here, cos FASTAPI MUST either return a JSONResponse or a StreamingResponse
    #When we use streaming we cant return any of the above. So we yield from the inner function and return the 
    #StreamingResponse from the outer fn. 
    async def event_stream():
        histories = request.app.state.histories
        #fetch the session_id from the QueryRequest
        session_id = payload.session_id
        #fetch the hostory of that session_id from session_id
        history = histories.get(session_id, [])[-8:]

        init_state = initial_state(payload.query, history=history)
        #Invoking the graph without the synth node.
        #request object contains a pointer to the state variable "state" ...
        final_state = request.app.state.graph.invoke(init_state)
        print(final_state.get("answer"))
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
            histories[session_id] = updated_history 
            yield f"event: clarify\ndata: {json.dumps({'clarifying_questions': questions})}\n\n"
            return
        

        #Handling the Out of Scope condition
        if final_state.get("out_of_scope"):
            answer = final_state.get("answer", "")
            updated_history = history + [{'role': 'user', 'content': payload.query},
                                     {'role': 'assistant', 'content': answer}]
            histories[session_id] = updated_history 
            yield f"event: scope_guard\ndata: {json.dumps({'answer': answer})}\n\n"
            return
    

        #Handling KB miss scenario
        grader = final_state.get("grader_output")
        if grader and grader.rerouting_decision == "detect":
            answer = final_state.get("answer", "")
            updated_history = history + [{'role': 'user', 'content': payload.query},
                                     {'role': 'assistant', 'content': answer}]
            histories[session_id] = updated_history 
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
        histories[session_id] = updated_history
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
                "modality":         analysis.modality         if analysis      else "N/A",
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


@app.post("/reset")
async def reset(request : Request, session_id: str = "default"):

    request.app.state.histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}