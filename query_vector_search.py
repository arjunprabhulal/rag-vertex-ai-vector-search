import os
from typing import List
import vertexai
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google import genai
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel



# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ID = "your-project-id"  # Replace with your GCP project ID
LOCATION = "us-central1"     # Google Cloud region
MODEL_NAME = "gemini-embedding-001"
EMBED_DIM = 3072
# Use the index and endpoint created by vector_search_index.py
INDEX_ENDPOINT = "projects/your-project-number/locations/us-central1/indexEndpoints/your-endpoint-id"
DEPLOYED_INDEX_ID = "gemini_vector_search_deployed"
API_ENDPOINT = "your-endpoint-id.us-central1-your-project-number.vdb.vertexai.goog"

# Route Google GenAI SDK to Vertex AI
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
genai_client = genai.Client()

# ---------------------------
# 1) Embed helpers (Gemini 001)
# ---------------------------
def embed_texts(texts: List[str], dim: int = EMBED_DIM) -> List[List[float]]:
    resp = genai_client.models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",  # good default for RAG docs/queries
            output_dimensionality=dim       # 3072 / 1536 / 768
        ),
    )
    return [e.values for e in resp.embeddings]

# ---------------------------
# 2) Direct Vector Search Query
# ---------------------------
def direct_vector_search(query_text: str, k: int = 10):
    # Embed the query text
    query_embedding = embed_texts([query_text])[0]
    
    # Configure Vector Search client
    client_options = {
        "api_endpoint": API_ENDPOINT
    }
    vector_search_client = aiplatform_v1.MatchServiceClient(
        client_options=client_options,
    )
    
    # Build FindNeighborsRequest object
    datapoint = aiplatform_v1.IndexDatapoint(
        feature_vector=query_embedding
    )
    
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=datapoint,
        neighbor_count=k
    )
    
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=INDEX_ENDPOINT,
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
        return_full_datapoint=True,  # Set to True to get full datapoints
    )
    
    # Execute the request
    response = vector_search_client.find_neighbors(request)
    
    return response


# ---------------------------
# 3) Generate with Gemini (grounded)
# ---------------------------
def answer_with_gemini(context_snippets: List[str], question: str) -> str:
    model = GenerativeModel("gemini-2.5-pro")  # or 2.5-flash for speed
    context = "\n\n".join(context_snippets)
    prompt = f"""Use only the context to answer. Provide a detailed and comprehensive response with thorough explanations. Include all relevant information from the context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}
Answer (provide a detailed explanation with at least 3-6 paragraphs):"""
    return model.generate_content(prompt).text


# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # This script only queries the Vector Search index created by vector_search_index.py
    
    # Example queries
    print("\n--- DIRECT VECTOR SEARCH QUERIES ---\n")
    
    # Query about Mixture of Experts (MoE)
    query = "What is Mixture of Experts (MoE) in large language models?"
    print(f"Query: {query}")
    
    # Retrieve relevant chunks from Vector Search
    response = direct_vector_search(query, k=5)
    
    print("\nRetrieved documents from Vector Search:")
    retrieved_chunks = []
    
    
    # Check if we have any results
    if not response.nearest_neighbors or not response.nearest_neighbors[0].neighbors:
        print("No results found in Vector Search.")
    else:
        # Process results
        for i, match in enumerate(response.nearest_neighbors[0].neighbors):
            datapoint_id = match.datapoint.datapoint_id
            distance = match.distance
            
            # Try to get content from restricts first (API approach)
            content = None
            if hasattr(match.datapoint, 'restricts') and match.datapoint.restricts:
                for restrict in match.datapoint.restricts:
                    if restrict.namespace == "content" and restrict.allow_list:
                        content = restrict.allow_list[0]
                        break
            
            # If not found in restricts, try crowding_tag
            if not content and hasattr(match.datapoint, 'crowding_tag') and match.datapoint.crowding_tag:
                crowding_attr = match.datapoint.crowding_tag.crowding_attribute
                if crowding_attr and crowding_attr != "0":
                    content = crowding_attr
            
            print(f"Document {i+1} - ID: {datapoint_id}, Distance: {distance:.4f}")
            
            if content:
                retrieved_chunks.append(content)
                # Handle different content types
                if isinstance(content, str):
                    print(f"Content: {content[:150]}...\n")
                else:
                    print(f"Content: {str(content)}\n")
            else:
                print("No content available in mapping\n")
        
        if retrieved_chunks:
            # Generate answer using the retrieved chunks
            print("\nGenerating answer based on retrieved chunks...")
            answer = answer_with_gemini(retrieved_chunks, query)
            print(f"Answer: {answer}")
        else:
            print("No text content found in the retrieved documents.")
    
    print("\nVector Search successfully retrieved and used relevant documents.")
