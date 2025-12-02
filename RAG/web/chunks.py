from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents , chunk_size =500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =chunk_size,
        chunk_overlap=chunk_overlap,
        length_function =len ,
        separators=["\n\n","\n"," ",""]
    )
    split_doc =text_splitter.split_documents(documents)
    
    print(f'\nsplit {len(documents)} document into {len(split_doc)} chunks \n')

    #if split_doc:
    #    print(f"\nExample chunk")
    #    print(f"Content:{split_doc[0].page_content[:10]}...........")
    #    print(f"Metadata :{split_doc[0].metadata}")
    return split_doc
