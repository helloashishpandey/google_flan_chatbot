import pandas as pd
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DataFrameLoader

app = Flask(__name__)

# Step 1: Load the CSV file
csv_file = "food_items.csv"
data = pd.read_csv(csv_file)
huggingface_token = "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"

# Create a dataframe with the relevant fields
dataframe = pd.DataFrame(data, columns=["Food Name", "Ingredients", "Image URL", "Description"])

# Step 2: Initialize HuggingFace model (google/flan-t5-small)
flan_model = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.7, "max_length": 200},
    huggingfacehub_api_token=huggingface_token  # Pass token explicitly
)

loader = DataFrameLoader(dataframe, page_content_column="Description")  # Use the 'Description' for content
documents = loader.load()


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

# Step 4: Build the LangChain Retrieval-based QA system
qa_chain = RetrievalQA.from_chain_type(
    llm=flan_model,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Step 5: Query function
def get_food_info(query):
    # Use the QA system to answer the query
    response = qa_chain.invoke(query)

    # Optional: Parse the source documents for additional info (e.g., image URL)
    sources = response['source_documents']
    if sources:
        # Extract image URL and additional details
        food_info = {
            "Description": sources[0].page_content,
            "Image URL": sources[0].metadata.get("Image URL")
        }
        return food_info
    return response

# Example User Query
query = "Tell me about Margherita Pizza"
result = get_food_info(query)

# Output
if result:
    print(f"Description: {result['Description']}")
    print(f"Image URL: {result['Image URL']}")
else:
    print("No matching food found!")

@app.route("/", methods=["GET"])
def info():
    return jsonify({"msg": "papa hoon me papa"})

@app.route("/query", methods=["POST"])
def query_food_info():
    try:
        # Get the user query from the request
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        # Get the result from LangChain QA system
        result = get_food_info(query)

        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "No matching food found!"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)