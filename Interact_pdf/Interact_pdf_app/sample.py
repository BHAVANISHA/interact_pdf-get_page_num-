# import os
# from django.http import JsonResponse
# from rest_framework.parsers import MultiPartParser, FormParser
# from .models import UploadedFile
# from .serializers import UploadedFileSerializer
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from rest_framework.generics import CreateAPIView
#
#
# class UploadedFileCreateAPIView(CreateAPIView):
#     serializer_class = UploadedFileSerializer
#     parser_classes = (MultiPartParser, FormParser)
#
#     def post(self, request, *args, **kwargs):
#         try:
#             # Create and validate the serializer
#             serializer = UploadedFileSerializer(data=request.data)
#             serializer.is_valid(raise_exception=True)
#
#             # Save the uploaded file(s) to the database
#             serializer.save()
#
#             # Retrieve data from the validated serializer
#             uploaded_files = UploadedFile.objects.filter(id=serializer.data['id'])
#             interact = uploaded_files.first()
#
#             # Use interact.file and interact.query as needed
#             pdf_files = [interact.file_upload.path]  # Keep it as a list for consistency
#             query = interact.query
#
#             # Set OpenAI API key
#             # os.environ["OPENAI_API_KEY"] = ''
#
#             # Process each uploaded file
#             all_results = []
#             for file_path in pdf_files:
#                 results_for_file = self.process_uploaded_file(file_path, query)
#                 all_results.extend(results_for_file)
#
#             if not all_results:
#                 return JsonResponse({"message": "No relevant documents found for the query."}, status=200)
#
#             return JsonResponse({"results": all_results}, status=200)
#
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#
#     def process_uploaded_file(self, file_path, query):
#         # Load the PDF
#         pdf_loader = PyPDFLoader(file_path)
#         documents = pdf_loader.load()
#
#         # Split documents into chunks
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#         docs = text_splitter.split_documents(documents)
#
#         # Use HuggingFace embeddings
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#
#         # Create FAISS index from documents
#         faiss_index = FAISS.from_documents(docs, embeddings)
#         faiss_index.save_local('index_store')
#
#         # Load FAISS index
#         vector_index = FAISS.load_local("index_store", embeddings)
#         retriever = vector_index.as_retriever(search_type="similarity")
#
#         # Retrieve relevant documents
#         retrieved_docs = retriever.get_relevant_documents(query)
#
#         document_names = []
#         for item in retrieved_docs:
#             print("Retrieved Item:", item)  # Add this print statement
#             if hasattr(item, 'metadata') and 'source' in item.metadata:
#                 source_path = item.metadata['source']
#                 if query.lower() in item.page_content.lower():
#                     page=item.metadata['page']
#                     pdf_name = os.path.basename(source_path)
#                     document_names.append('page'+str(page)+pdf_name)
#
#         print("Document Names:", document_names)  # Add this print statement
#         return document_names
#
# ===========================================================load pdf=======================================================================
# import os
# from django.http import JsonResponse
# from rest_framework.parsers import MultiPartParser, FormParser
# from .models import UploadedFile
# from .serializers import UploadedFileSerializer
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from rest_framework.generics import CreateAPIView
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
#             os.environ["OPENAI_API_KEY"] = ''
#
#             # Use interact.file and interact.query as needed
#             pdf_files = [interact.file_upload.path]  # Keep it as a list for consistency
#             query = interact.query
#
#             # Process each uploaded file
#             all_results = ''
#             for file_path in pdf_files:
#                 results_for_file = self.process_uploaded_file( file_path, query )
#                 all_results+= results_for_file
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
#         pdf_loader = PyPDFLoader( file_path )
#         documents = pdf_loader.load()
#
#         # Use HuggingFace embeddings
#         embeddings = HuggingFaceEmbeddings( model_name="all-MiniLM-L6-v2" )
#
#         # Create FAISS index from documents
#         faiss_index = FAISS.from_documents( documents, embeddings )
#         faiss_index.save_local( 'index_store' )
#
#         # Load FAISS index
#         vector_index = FAISS.load_local( "index_store", embeddings )
#         retriever = vector_index.as_retriever( search_type="similarity" )
#
#         # Use Langchain's RetrievalQA
#         qa_interface = RetrievalQA.from_chain_type(
#             llm=ChatOpenAI( model_name='gpt-3.5-turbo-16k' ),
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True,
#         )
#
#         # Query the QA interface
#         response = qa_interface( query )
#
#         # Create a list to store retrieved documents
#         retrieved_docs = []
#
#         # Iterate through each document and check if it contains the query
#         for doc in documents:
#             content = doc.page_content.lower()
#             if query.lower() in content:
#                 retrieved_docs.append( doc )
#
#         # Create a dictionary to store document names
#         document_names = {}
#         for item in retrieved_docs:
#             source_path = item.metadata['source']
#             page = item.metadata['page']
#             pdf_name = os.path.basename( source_path )
#
#             if pdf_name not in document_names:
#                 document_names[pdf_name] = []
#
#             document_names[pdf_name].append( page )
#
#         print( "Query:", query )
#         print( "Number of Retrieved Documents:", len( retrieved_docs ) )
#         print( "Document Names:", document_names )
#
#         return response['result']
# =======================================================================================================
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
            return_source_documents=True,
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
