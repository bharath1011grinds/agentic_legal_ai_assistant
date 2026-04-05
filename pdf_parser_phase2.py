import os
import requests
import pymupdf4llm
import pymupdf
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def download_arxiv_pdf(arxiv_id : str, save_path: str = "data/papers/") -> str:

    os.makedirs(save_path, exist_ok=True)
    pdf_path = os.path.join(save_path,f'{arxiv_id}.pdf')

    #skip if that pdf is already downloaded.
    if os.path.exists(pdf_path):
        print(f"File {arxiv_id} already exists...")
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f" Downloading {arxiv_id}...")

    #30 second timeout
    response = requests.get(url, timeout=30)
    #raise exception if status is 404 or 500
    response.raise_for_status()

    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Saved to {pdf_path}")
    return pdf_path


def extract_text_as_markdown(pdf_path : str)-> str:

    print(f"Extracting text from {Path(pdf_path).name}...")

    #the whole point of using pymupdf4llm - able to convert text to markdown
    #remove repeating headers like journal name and footers like page number, that break the flow of text.
    #also helps convert tables into markdown format and parse them as text
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        # Exclude repetitive header/footer text (page numbers, journal name)
        header=False,
        footer=False,
    )

    return md_text


def extract_images(pdf_path:str, output_dir:str = 'data/images/', min_size_kb : int = 10) -> list[dict]:

    #Create the dir if its the first time executing the function.
    os.makedirs(output_dir, exist_ok=True)
    #Converts the path_string to Path object and retrieves just the name of the file without the file extension(.pdf)
    paper_name = Path(pdf_path).stem
    extracted = []

    pdf = pymupdf.open(pdf_path)

    for page_num in range(len(pdf)):
        #get Documents object for each page.
        page = pdf[page_num]
        #get a list of all the images in that page. returns a list of tuples, tuple contains metadata about the image
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            # unique identifier for each image    
            xref = img_info[0]
            #extract a dict corresponding to the image from the pdf via the unique id
            base_image = pdf.extract_image(xref=xref)

            #extract the image bytes
            image_bytes = base_image['image']
            ext = base_image['ext'] #get the extension from the dict...

            #ignore small unrelated images, decorative elements, scientific figures etc.
            size_kb = len(image_bytes)/1024 #get the image size in kbs
            if size_kb<min_size_kb:
                continue
            
            #save the image
            img_filename = f"{paper_name}_p{page_num+1}_img{img_idx+1}.{ext}"
            img_path = os.path.join(output_dir, img_filename)

            with open(img_path, 'wb') as f: #write as bytes
                f.write(image_bytes)
            
            extracted.append({
                "path":     img_path,
                "page":     page_num + 1,
                "size_kb":  round(size_kb, 1),
                "format":   ext,
                "paper_id": paper_name
            })

    pdf.close()
   
    print(f"Extracted {len(extracted)} figures from {Path(pdf_path).name}")
    return extracted


def extract_tables(pdf_path : str) -> list[dict]:

    # WHY INDEX TABLES SEPARATELY?
    #A query like "What BLEU score did the paper achieve?" needs the table data,not the surrounding prose. 
    #Indexing tables as separate Documents means the retriever can surface them directly.

    pdf = pymupdf.open(pdf_path)
    paper_name = Path(pdf_path).stem
    tables = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]

        tables = page.find_table()

        for tbl_idx, table in enumerate(tables.tables):
            try:
                # Convert to list of rows
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue

                # Build Markdown table string
                #NOTE: Need to go through the markdown table logic
                header = "| " + " | ".join(str(c) for c in rows[0]) + " |"
                divider = "| " + " | ".join(["---"] * len(rows[0])) + " |"
                body = "\n".join(
                    "| " + " | ".join(str(c) for c in row) + " |"
                    for row in rows[1:]
                )
                md_table = f"{header}\n{divider}\n{body}"

                tables.append({
                    "content":  md_table,
                    "page":     page_num + 1,
                    "table_id": tbl_idx + 1,
                    "paper_id": paper_name
                })

            except Exception:
                continue
    pdf.close()

    print(f"Extracted {len(tables)} tables from {Path(pdf_path).name}")
    return tables



#BUILD LANGCHAIN DOCUMENTS:
def pdf_to_documents(pdf_path:str, metadata:dict = {})-> list[Document]:

    #Two stage splitting done, better than phase 1
    #1. Markdownheadertextsplitter - convert the text to markdown and split them at different markdown headers, 
    # like different sections boundaries(##result, ##introduction etc..)
    #2. Recursivecharactertextsplitter - If the chunks are still too long after MDheadertextsplit, use this to chunk them.

    #This two stage chunking ensures, semantic bounding(no mid-sentence or mid-topic breakage) and size-boounding too.

    documents = []

    #get the text as markdown
    md_text = extract_text_as_markdown(pdf_path=pdf_path)
    #The below splits the markdown text on the 3 boundaries, #, ##, ###.
    #The second part of the tuple, "section", "subsection" and "subsubsection" become metadata tags for the text that splits the chunks.
    #eg. if there is a tag #Introduction, there will be a metadata tag(chunk-wise metadata) added as 
    # {"section" : "Introduction"}
    #if strip_headers = True, the #Introduction will be only in metadata and will be removed from the chunk, it will 
    # be present as a part of the chunk if strip_headers = False 
    header_splitter = MarkdownHeaderTextSplitter([("#", "section"), ("##", "subsection"), ("###", "subsubsection")],
                                                 strip_headers=False)#Keep headers to preserve more context
    
    header_chunks = header_splitter.split_text(md_text)

    #2nd level of chunking after header chunking, if very big chunks(>600) are found.
    #Follows a splitting hierarchy, specified in the "separators" argument.
    #Add \n| to hierarchy so that, we ensure table rows are intact and if chunking is necessary, it happens at the end of row
    #before having to do it mid-row
    char_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 60, 
                                                   separators=["\n\n", "\n|","\n", ". ", " "])

    text_chunks = char_splitter.split_documents(header_chunks)

    #attach the pdf metadata to every chunk...
    for chunk in text_chunks:
        chunk.metadata.update({
            **metadata, 
            'source' : pdf_path,
            'content_type' : 'text'
        }
        )

    documents.extend(text_chunks)

    #Commenting the below code because, we are already proecssing the tables as we process text, and this is redundant.
    #Also, tables are not of much  use if we store them without the context words
    '''
    #---TABLES---

    tables = extract_tables(pdf_path=pdf_path)#defined above

    for tbl in tables:
        tbl_doc = Document(page_content=f"Table (page {tbl['page']}:\n {tbl['content']}",
                            metadata = {**metadata, 'content_type':'table', 'source':pdf_path, 'page':tbl['page']})
        
        documents.append(tbl_doc)

    print(f"Created {len(documents)} documents ({len(text_chunks)} text + {len(tables)} tables)")
    '''
    print(f"Created {len(documents)} documents ({len(text_chunks)})") 
    return documents
     


#Full Pipeline
def process_arxiv_paper(arxiv_id:str, extra_metadata: dict = {}) -> list[Document]:

    print(f"\n Processing paper: {arxiv_id}")

    pdf_path = download_arxiv_pdf(arxiv_id)
    images   = extract_images(pdf_path)  # saved to disk, to be captioned in later...

    docs = pdf_to_documents(pdf_path=pdf_path, metadata={**extra_metadata, 'arxiv_id':arxiv_id,
                                                          'url':f"https://arxiv.org/abs/{arxiv_id}"})
    
    return docs, images


if __name__ =='__main__':

    # Testing with the original "Attention Is All You Need" paper
    docs, images = process_arxiv_paper(
        "1706.03762",
        extra_metadata={"title": "Attention Is All You Need"}
    )

    print(f"\n Summary:")
    print(f"  Documents : {len(docs)}")
    print(f"  Images    : {len(images)}")
    print(f"\nSample chunk:")
    print(docs[0].page_content[:300])
    print(f"\nMetadata: {docs[0].metadata}")

                                                        


