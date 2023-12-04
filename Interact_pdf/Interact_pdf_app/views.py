# import os
# from django.http import JsonResponse
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.utils import cosine_similarity
# from langchain.vectorstores.chroma import Chroma
# from rest_framework.parsers import MultiPartParser, FormParser
# from sqlalchemy.testing import db
#
# from .models import UploadedFile
# from .serializers import UploadedFileSerializer
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from rest_framework.generics import CreateAPIView
# import docsearch  # Import the docsearch library
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import create_qa_with_sources_chain
#
#
# class UploadedFileCreateAPIView( CreateAPIView ):
#     serializer_class = UploadedFileSerializer
#     parser_classes = (MultiPartParser, FormParser)
#
#     def post(self, request, *args, **kwargs):
#         try:
#             # Create and validate the serializer
#             serializer = UploadedFileSerializer( data=request.data )
#             serializer.is_valid( raise_exception=True )
#
#             # Save the uploaded file(s) to the database
#             serializer.save()
#
#             # Retrieve data from the validated serializer
#             uploaded_files = UploadedFile.objects.filter( id=serializer.data['id'] )
#             interact = uploaded_files.first()
#             os.environ["OPENAI_API_KEY"] = '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'
#             # Use interact.file and interact.query as needed
#             pdf_files = [interact.file_upload.path]  # Keep it as a list for consistency
#             query = interact.query
#
#             # Process each uploaded file
#             all_results = []
#             for file_path in pdf_files:
#                 results_for_file = self.process_uploaded_file( file_path, query )
#                 all_results.extend( results_for_file )
#
#             if not all_results:
#                 return JsonResponse( {"message": "No relevant documents found for the query."}, status=200 )
#
#             return JsonResponse( {"results": all_results}, status=200 )
#
#         except Exception as e:
#             return JsonResponse( {"error": str( e )}, status=500 )
#
#     def process_uploaded_file(self, file_path, query):
#         # Load the PDF
#
#         # pdf_loader = PyPDFLoader( file_path )
#         # documents = pdf_loader.load()
#         # print( len( documents ) )
#         loader = PyPDFLoader( file_path )
#
#         pages = loader.load_and_split()
#         print( len( pages ) )
#         # Use HuggingFace embeddings
#         embedding_function = SentenceTransformerEmbeddings( model_name="all-MiniLM-L6-v2" )
#         db = Chroma.from_documents( pages, embedding_function )
#         llm_src = ChatOpenAI( temperature=0, model="gpt-3.5-turbo-0613" )
#
#         qa_chain = create_qa_with_sources_chain( llm_src )
#
#         doc_prompt = PromptTemplate(
#             template="Content: {page_content}\nSource: {source} \n Page:{page}",  # look at the prompt does have page#
#             input_variables=["page_content", "source", "page"],
#         )
#
#         final_qa_chain = StuffDocumentsChain(
#             llm_chain=qa_chain,
#             document_variable_name='context',
#             document_prompt=doc_prompt,
#         )
#         retrieval_qa = RetrievalQA(
#             retriever=db.as_retriever(),
#             combine_documents_chain=final_qa_chain
#         )
#
#         # query = query
#
#         response = retrieval_qa.run( query )
#         result = response['result']
#         retrieved_docs = []
#         print( result )
#
#         # Iterate through each document and check if it contains the query
#         # Iterate through each document and check if it contains the query
#         for page in pages:
#             content = page.page_content.lower()
#             if query.lower() in content:
#                 retrieved_docs.append( page )
#
#         # Create a list to store document names and corresponding page numbers
#         # Create a list to store document names and corresponding page numbers
#         document_names_list = []
#
#         # Iterate through each retrieved document and check if it contains the result
#         # Iterate through each retrieved document and check if it contains the result
#         for item in retrieved_docs:
#             print( "Item:", item )
#
#             # Check if item has 'metadata' attribute
#             if hasattr( item, 'metadata' ):
#                 print( "Metadata:", item.metadata )
#
#                 # Check if 'metadata' is a dictionary
#                 if isinstance( item.metadata, dict ):
#                     content = item.page_content.lower()
#
#                     # Check if the result is in the document content
#                     if result.lower() in content:
#                         source_path = item.metadata.get( 'source', '' )
#                         page = item.metadata.get( 'page', '' )
#                         pdf_name = os.path.basename( source_path )
#
#                         # Store the document name and page number in a dictionary
#                         document_names = {
#                             'pdf_name': pdf_name,
#                             'page': page,
#                         }
#
#                         document_names_list.append( document_names )
#
#                         # Print information for each retrieved document
#                         print( "Document Name:", pdf_name )
#                         print( "Page Number:", page )
#                 else:
#                     print( "'metadata' is not a dictionary." )
#             else:
#                 print( "Item has no 'metadata' attribute." )
#
#         # Check if any relevant documents were found
#         if not document_names_list:
#             print( "No relevant documents found for the query." )
#
#         # Print the final result
#         print( "Query:", query )
#         print( "Result:", result )
#         print( "Document Names List:", document_names_list )
#
#         # Return the document_names_list or use it as needed in your application
#         return document_names_list
# ==========================================================================================================================================



import os
from django.http import JsonResponse
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from rest_framework.parsers import MultiPartParser, FormParser
from .models import UploadedFile
from .serializers import UploadedFileSerializer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from rest_framework.generics import CreateAPIView
import docsearch  # Import the docsearch library


class UploadedFileCreateAPIView( CreateAPIView ):
    serializer_class = UploadedFileSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Create and validate the serializer
            serializer = UploadedFileSerializer( data=request.data )
            serializer.is_valid( raise_exception=True )

            # Save the uploaded file(s) to the database
            serializer.save()

            # Retrieve data from the validated serializer
            uploaded_files = UploadedFile.objects.filter( id=serializer.data['id'] )
            interact = uploaded_files.first()
            os.environ["OPENAI_API_KEY"] = ''
            # Use interact.file and interact.query as needed
            pdf_files = [interact.file_upload.path]  # Keep it as a list for consistency
            query = interact.query

            # Process each uploaded file
            all_results = {}
            for file_path in pdf_files:
                results_for_file = self.process_uploaded_file( file_path, query )
                all_results.update( results_for_file )

            if not all_results:
                return JsonResponse( {"message": "No relevant documents found for the query."}, status=200 )

            return JsonResponse( {"results": all_results}, status=200 )

        except Exception as e:
            return JsonResponse( {"error": str( e )}, status=500 )


    def process_uploaded_file(self, file_path, query):
        # Load the PDF
        pdf_loader = PyPDFLoader( file_path )
        documents = pdf_loader.load()
        print( len( documents ) )

        # Use HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2" )
        vector_index = FAISS.load_local( "index_store", embeddings )
        retriever = vector_index.as_retriever( search_type="similarity" )

        # Use Langchain's RetrievalQA
        qa_interface = RetrievalQA.from_chain_type(
            llm=ChatOpenAI( model_name='gpt-3.5-turbo-16k' ),
            chain_type="stuff",
            retriever=retriever,
            # return_source_documents=True,
        )

        # Query the QA interface
        response = qa_interface( query )
        result = response['result']
        # Create list to store retrieved documents
        retrieved_docs = []

        # Iterate through each document and check if it contains the query
        for doc in documents:
            content = doc.page_content.lower()
            if query.lower() in content:
                retrieved_docs.append( doc )

        document_names = {}
        for item in retrieved_docs:
            source_path = item.metadata['source']
            page = item.metadata['page']
            pdf_name = os.path.basename( source_path )

            if pdf_name not in document_names:
                document_names[pdf_name] = []

            document_names[pdf_name].append( page )
        print( "Query:", query )
        print( "Number of Retrieved Documents:", len( retrieved_docs ) )
        # for item in retrieved_docs:
        #     print("Page Content:", item.page_content)
        document_names['answer'] = result
        print("Document Names:", document_names)
        return document_names
