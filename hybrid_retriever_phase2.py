import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_cross_encoder = None

def get_cross_encoder(model_name : str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    global _cross_encoder

    if _cross_encoder is None:
        print(f"Loading cross-encoder '{model_name}'...")
        #max_length refers to the max sequence length of the chunk and query combined...
        _cross_encoder = CrossEncoder(model_name = model_name, max_length=512)
        print("Cross-encoder loaded")
    return _cross_encoder


def build_bm25_index(documents:list[Document]) -> tuple[BM25Okapi, list[Document]]:

    #BM25 vs tf-idf : BM25 has a saturation on term frequency saturation, score doesnt increase after a point where the word gets too repititive
    #BM25 also normalizes the doc length, a term in a short doc is weighted more than in a long doc per occurance.
   
    tokenized_corpus = [doc.page_content.lower().split() for doc in documents] #can use word-tokenize library here...

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, documents



#Create a Langchain BaseRetriever Subclass to apply custom logic and use it "WHEREVER A RETRIEVER IS REQUIRED..."
#This inheritance makes the pipeline.py work smoothly even after changing the retriever,
#with just 1 line of change in the code.(retriever initialization)
class HybridRetriever(BaseRetriever):

#NOTE: the HybridRetiever should have been written with an __init__ to use instance variables instead of class variables


#Required Pydantic fields:
#exclude = True is used to omit the fields and values from being added to exports like, json.dump().
#The fields still remain accessible within the code.
    vectorstore:      FAISS           = Field(exclude=True)
    documents:        List[Document]  = Field(exclude=True)
    bm25:             BM25Okapi       = Field(exclude=True)
    k:                int             = Field(default=5)
    bm25_candidates:  int             = Field(default=20)
    faiss_candidates: int             = Field(default=20)

    class Config:#Nested class, applies changes just within this class and its objects...
        arbitrary_types_allowed = True  # needed for non-Pydantic types like BM25Okapi

    #BM25 retrieval function
    def _bm25_retrieve(self, query:str) -> List[Document]: 
        
        #Tokenize the query as we did for the corpus
        tokenized_query = query.lower().split()

        #bm25 field is a BM25OKapi field that contains the embedded corpus, will be passed later...
        scores = self.bm25.get_scores(tokenized_query)

        #sort the array, get the indices from min to max, reverse index for max scores, slice the required number of scores...
        top_indices = np.argsort(scores)[::-1][:self.bm25_candidates]#note this returns just the indices of the scores...

        #get the docs corresponding to the max non-zero scores
        results = [self.documents[i] for i in top_indices if scores[i]>0]

        return results
    
    #FAISS MMR(max marginal relevance) retrieval
    def _faiss_retrieve(self, query:str) -> List[Document]:

        #MMR retrieval considers a larger pool before selecting the dissimilar chunks from the retrieved items...
        #fetch_k = 2*faiss_candidates
        #formula : mmr_score = lambda*(similarity to query) - (1-lambda)*(similarity to the selected chunks);
        # lambda = 0.5 by default in langchain, equal importance to similarity and diversity
        #candidates returned = faiss_candidates
        results = self.vectorstore.max_marginal_relevance_search(query=query, fetch_k=2*self.faiss_candidates,
                                                                 k = self.faiss_candidates)
        return results
    
    #Deduplication of the combined search results - Deduplicate(MMR + BM25)
    def _deduplicate(self, docs : List[Document]) -> List[Document]:
        #deduplication is done to save compute for the crossencoder...
        seen = set()
        unique = []
        for doc in docs:
            #using hash to identify the same chunks
            hashed = hash(doc.page_content)
            if hashed not in seen:
                seen.add(hashed)
                unique.append(doc)
        return unique
        
    #CrossEncoder Reranking...
    def _rerank(self, query : str, candidates : List[Document]) -> List[Document]:

        results =[]
        if not candidates:
            return []
        cross_encoder = get_cross_encoder()
        #The order matters, (query, document) because, ms-marco was trained in the same order, (query, document), it can confuse the model
        #if we change the order. Also, if the doc is too long, it will be truncated even before seeing the query because of max_length.
        #attention mask differs for the two, weight initialization differs, always follow order, (query, doc)
        pairs = [(query, doc.page_content) for doc in candidates]

        scores = cross_encoder.predict(pairs)

        ranked = sorted(zip(candidates, scores), key=lambda x:x[1], reverse=True)
        
        #Fetch topk reranked
        for doc, score in ranked[:self.k]:
            doc.metadata['rerank_score'] = round(float(score), 4)
            results.append(doc)
        
        return results

    #Main Retrieval method
    #The _get_relevant_documents function is mandatory for any class inheriting the BaseRetriever class..
    #This method gets internally called when this retriever is used inside create_chain_retriver, in the pipeline.py files...
    def _get_relevant_documents(self, query) -> List[Document]:

        #Retrieve the data parallely
        bm25_results = self._bm25_retrieve(query=query)
        mmr_results = self._faiss_retrieve(query=query)

        #combine results
        combined = bm25_results+mmr_results
        
        #deduplicate
        deduplicated = self._deduplicate(combined)

        #rerank
        reranked = self._rerank(query=query, candidates=deduplicated)

        print(f"\nRetrieval stats:")
        print(f"   BM25 candidates  : {len(bm25_results)}")
        print(f"   FAISS candidates : {len(mmr_results)}")
        print(f"   After dedup      : {len(deduplicated)}")
        print(f"   Final (reranked) : {len(reranked)}")
        if reranked:
            print(f"   Top rerank score : {reranked[0].metadata.get('rerank_score', 'N/A')}")
        
        return reranked
    

#FACTORY FUNCTION - build retriever from Index

def build_hybrid_retriever(
    index_path:       str = "vectorstore/",
    k:                int = 5,
    bm25_candidates:  int = 20,
    faiss_candidates: int = 20
    ) -> HybridRetriever :

    #This function builds the hybridretriever and returns it. We neeed to call just this function in the piepline.py

    print("Building hybrid retriever...")

    # Load FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Extract all documents from FAISS for BM25
    # FAISS stores (embedding_id -> document) in its docstore
    #The vector metadata are stored in vectorstore.docstore. 
    #docstore is a dict like object and the dict can be accessed via a private attribute, _dict.
    #_dict.values() gets only all the values[actual Document objects] from the dictionary.
    documents = list(vectorstore.docstore._dict.values())
    print(f"Loaded {len(documents)} documents from FAISS")

    bm25, docs = build_bm25_index(documents=documents)

    #NOTE: the HybridRetiever should have been written with an __init__ to use instance variables instead of class variables
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        documents=docs,
        bm25=bm25,
        k=k,
        bm25_candidates=bm25_candidates,
        faiss_candidates=faiss_candidates
    )

    print("HybridRetriever ready")
    return retriever

