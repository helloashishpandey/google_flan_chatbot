import streamlit as st
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DataFrameLoader

# Streamlit page configuration
st.set_page_config(page_title="Food Info Finder", page_icon="🍕", layout="wide")

# App title
st.title("Food Info Finder 🍴")

# File uploader
csv_file = st.file_uploader("Upload a CSV file with food details:", type=["csv"])

if csv_file:
    # Step 1: Load CSV data
    st.success("File uploaded successfully! Processing...")
    data = pd.read_csv(csv_file)
    
    # Check if required columns are present
    required_columns = ["Food Name", "Ingredients", "Image URL", "Description"]
    if all(col in data.columns for col in required_columns):
        dataframe = pd.DataFrame(data, columns=required_columns)
        
        # Step 2: Initialize HuggingFace model
        huggingface_token = st.text_input("Enter your HuggingFace API Token:")
        if huggingface_token:
            flan_model = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature": 0.7, "max_length": 200},
                huggingfacehub_api_token=huggingface_token
            )
            
            # Step 3: Prepare the data as retrievable documents
            loader = DataFrameLoader(dataframe, page_content_column="Description")
            documents = loader.load()
            print(documents)
            
            # Embed the data for querying
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(documents, embedding_model)
            
            # Step 4: Build the LangChain Retrieval-based QA system
            qa_chain = RetrievalQA.from_chain_type(
                llm=flan_model,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Step 5: Query section
            st.subheader("Query Food Information")
            query = st.text_input("Ask about a food item (e.g., 'Tell me about Margherita Pizza'):")
            
            if query:
                # Query function
                def get_food_info(query):
                    response = qa_chain.invoke(query)
                    sources = response.get('source_documents', [])
                    if sources:
                        food_info = {
                            "Description": sources[0].page_content,
                            "Image URL": sources[0].metadata.get("Image URL")
                        }
                        return food_info
                    return response
                
                # Get and display result
                with st.spinner("Fetching information..."):
                    result = get_food_info(query)
                    
                if result:
                    st.write("### Food Description")
                    st.write(result.get("Description", "No description available"))
                    image_url = result.get("Image URL", None)
                    if image_url:
                        st.image(image_url, caption="Food Image")
                    else:
                        st.write("No image URL available for this item.")
                else:
                    st.error("No matching food found!")
    else:
        st.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")
else:
    st.info("Upload a CSV file to begin.")
