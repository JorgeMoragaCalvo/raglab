import asyncio

import streamlit as st
import requests
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader
import os
import tempfile
import arxiv
import time
import json

from typing import List, Dict, Optional
from persistence import PersistenceManager, create_persistence_manager

load_dotenv('.env')

# styles configuration
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd){
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border: 1px solid #3A3A3A !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even){
        background-color: #2A2A2A !important;
        color: #F0F0F0 !important;
        border: 1px solid #404040 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    .paper-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3A3A3A;
        margin: 10px 0;
    }
    .subject-tag {
        background-color: #00FFAA;
        color: #000;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    LANGUAGE_MODEL = GoogleGenerativeAI(model="gemini-2.5-flash")

PROMPT_TEMPLATE = """
You are an expert research assistant specialized in analyzing research papers. 
Use the provided context from the research paper to answer the query accurately.
If the information is not in the context, clearly state that you don't know.
Provide precise answers about methodologies, results, conclusions, and technical details.

Query: {user_query}
Context from paper: {document_context}

Answer:
"""

class SemanticScholarSearcher:
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {
            'User-Agent': 'Research-Assistant/1.0',
            'Accept': 'application/json'
        }

        # Agregar API key si está disponible (aumenta límites)
        if api_key:
            self.headers['x-api-key'] = api_key

        # Control de rate limiting
        self.last_request_time = 0
        self.min_interval = 1.0  # Mínimo 1 segundo entre requests
        self.max_retries = 3

    def _wait_for_rate_limit(self):
        """Implementa rate limiting básico"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request_with_backoff(self, url: str, params: dict = None, timeout: int = 30):
        """Hace request con manejo de 429 y backoff exponencial"""
        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()

            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=timeout)

                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    # Rate limited - esperar más tiempo
                    wait_time = (2 ** attempt) * 2  # Backoff exponencial: 2, 4, 8 segundos
                    st.warning(
                        f"Rate limit reached. Waiting {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f'Error on Semantic Scholar API: {response.status_code}')
                    return response

            except requests.exceptions.Timeout:
                st.error(f"Timeout en intento {attempt + 1}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except Exception as e:
                st.error(f'Error on request: {e}')
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        return None

    def search_papers(self, query: str, max_results: int = 10, fields: List[str] = None) -> List[Dict]:
        """Search papers on Semantic Scholar con rate limiting"""
        if fields is None:
            fields = ['paperId', 'title', 'abstract', 'authors', 'year', 'venue',
                      'citationCount', 'influentialCitationCount', 'url', 'openAccessPdf']

        search_url = f"{self.base_url}/paper/search"
        params = {
            'query': query,
            'limit': min(max_results, 100),  # Semantic Scholar max is 100
            'fields': ','.join(fields)
        }

        response = self._make_request_with_backoff(search_url, params)

        if response and response.status_code == 200:
            data = response.json()
            papers = []

            for paper_data in data.get('data', []):
                paper_info = {
                    'id': paper_data.get('paperId', ''),
                    'title': paper_data.get('title', 'Untitled'),
                    'authors': [author['name'] for author in paper_data.get('authors', [])],
                    'abstract': paper_data.get('abstract', 'Abstract not available'),
                    'year': str(paper_data.get('year', 'Year unknown')),
                    'venue': paper_data.get('venue', 'Venue unknown'),
                    'citations': paper_data.get('citationCount', 0),
                    'influential_citations': paper_data.get('influentialCitationCount', 0),
                    'url': paper_data.get('url', ''),
                    'pdf_url': paper_data.get('openAccessPdf', {}).get('url', '') if paper_data.get(
                        'openAccessPdf') else '',
                    'source': 'Semantic Scholar',
                    'semantic_data': paper_data
                }
                papers.append(paper_info)
            return papers

        return []

    def get_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Obtiene detalles de un paper con rate limiting"""
        url = f'{self.base_url}/paper/{paper_id}'
        params = {
            'fields': 'title,abstract,authors,year,venue,citationCount,influentialCitationCount,references,citations,tldr,fieldsOfStudy'
        }

        response = self._make_request_with_backoff(url, params)

        if response and response.status_code == 200:
            return response.json()
        return None

    def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download PDF con rate limiting"""
        if not pdf_url:
            return None

        response = self._make_request_with_backoff(pdf_url)

        if response and response.status_code == 200:
            return response.content
        return None

    def batch_search(self, queries: List[str], max_results_per_query: int = 10) -> Dict[str, List[Dict]]:
        """Búsqueda en lote con rate limiting automático"""
        results = {}
        total_queries = len(queries)

        for i, query in enumerate(queries):
            st.info(f"Procesando consulta {i + 1}/{total_queries}: {query[:50]}...")
            results[query] = self.search_papers(query, max_results_per_query)

            # Pausa adicional entre consultas en lote
            if i < total_queries - 1:
                time.sleep(1)

        return results

class ArxivSearcher:
    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(self, query, max_results=10, sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance):
        """Search papers on arXiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )

            papers = []
            for result in self.client.results(search):
                paper_info = {
                    'id': result.entry_id.split('/')[-1],  # Extraer ID de arXiv
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'updated': result.updated.strftime('%Y-%m-%d') if result.updated else None,
                    'categories': result.categories,
                    'primary_category': result.primary_category,
                    'pdf_url': result.pdf_url,
                    'entry_id': result.entry_id,
                    'doi': result.doi,
                    'journal_ref': result.journal_ref,
                    'comment': result.comment,
                    'source': 'arXiv',
                    'year': result.published.strftime('%Y')
                }
                papers.append(paper_info)

            return papers

        except Exception as e:
            st.error(f'Error searching on arXiv: {e}')
            return []

    def download_pdf(self, pdf_url):
        """Downloading PDF from arXiv"""
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            st.error(f'Error downloading PDF: {e}')

class PaperProcessor:
    def __init__(self):
        pass

    def process_with_arxiv_loader(self, arxiv_id):
        """Using ArxivLoader from LangChain to process the paper"""
        try:
            # ArxivLoader descarga y procesa automáticamente
            loader = ArxivLoader(
                query=arxiv_id,
                load_max_docs=1,
                load_all_available_meta=True
            )

            documents = loader.load()

            if documents:
                return {
                    'documents': documents,
                    'title': documents[0].metadata.get('Title', 'Untitled'),
                    'abstract': documents[0].metadata.get('Summary', 'No abstract'),
                    'authors': documents[0].metadata.get('Authors', 'No authors'),
                    'published': documents[0].metadata.get('Published', 'No date'),
                    'full_text': documents[0].page_content
                }
            return None

        except Exception as e:
            st.error(f'Error with ArxivLoader: {e}')
            return None

    def process_with_pypdf(self, pdf_content):
        """Process PDF with PyPDFLoader as alternative"""
        try:
            # Check PDF header
            if not pdf_content or not pdf_content.startswith(b'%PDF'):
                st.error(f"Downloaded file is not a valid PDF. First bytes: {pdf_content[:20]}")
                return None
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name

            # Using PyPDFLoader
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            # Cleaning a temporary file
            os.unlink(temp_file_path)

            if documents:
                full_text = "\n\n".join([doc.page_content for doc in documents])
                return {
                    'documents': documents,
                    'full_text': full_text,
                    'title': 'Extraído de PDF',
                    'abstract': self.extract_abstract_from_text(full_text)
                }
            return None

        except Exception as e:
            st.error(f'Error processing PDF: {type(e).__name__}{e}')
            return None

    def extract_abstract_from_text(self, text):
        """Extract abstract from the full text"""
        lines = text.split('\n')
        abstract_content = []
        in_abstract = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect start of the abstract
            if any(keyword in line.lower() for keyword in ['abstract', 'summary']):
                in_abstract = True
                if len(line) > 20: # Si la línea tiene contenido además de "Abstract"
                    abstract_content.append(line)
                continue

            # Detect the end of the abstract
            if in_abstract and any(keyword in line.lower() for keyword in ['introduction', '1.', 'keywords', 'key words']):
                break

            if in_abstract:
                abstract_content.append(line)
                if len(' '.join(abstract_content)) > 500: # limiting size
                    break

        return ' '.join(abstract_content) if abstract_content else text[:300]

# def create_vector_store(documents):
#     """Creates a vector store with documents"""
#     if not documents or not GOOGLE_API_KEY:
#         return None
#
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=300,
#         add_start_index=True
#     )
#
#     splits = text_splitter.split_documents(documents)
#     vectorstore = FAISS.from_documents(splits, EMBEDDING_MODEL)
#     return vectorstore

def create_vector_store(documents, paper_id=None, persistence_manager=None):
    if not documents or not GOOGLE_API_KEY:
        return None

    if persistence_manager:
        return documents
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            add_start_index=True
        )

        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, EMBEDDING_MODEL)
        return vectorstore

def generate_answer(user_query, context_documents):
    """Generates answer using Google Gemini"""
    if not GOOGLE_API_KEY:
        return "Error: Gemini API KEY not configured"

    context_text = "\n\n".join([doc.page_content for doc in context_documents])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | LANGUAGE_MODEL

    return chain.invoke({
        "user_query": user_query,
        "document_context": context_text
    })


class ImprovedSemanticScholarSearcher(SemanticScholarSearcher):
    """Extensión de SemanticScholarSearcher con mejor manejo de PDFs"""

    def validate_pdf_content(self, content: bytes) -> bool:
        """Valida si el contenido descargado es realmente un PDF"""
        if not content:
            return False

        # Valid PDF start with %PDF
        if content.startswith(b'%PDF'):
            return True

        # Verificar si es una respuesta JSON de error
        try:
            content_str = content.decode('utf-8')[:100]  # Primeros 100 caracteres
            if content_str.strip().startswith('{'):
                # Es JSON, probablemente un error
                json.loads(content_str)
                return False
        except:
            pass

        return False

    def download_pdf_with_validation(self, pdf_url: str) -> Optional[bytes]:
        """Descarga PDF con validación mejorada"""
        if not pdf_url:
            st.warning("⚠️ No URL PDF available")
            return None

        try:
            st.info(f"🔍 Trying to download PDF from: {pdf_url[:60]}...")

            # Headers más completos para evitar bloqueos
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            response = requests.get(pdf_url, headers=headers, timeout=30, allow_redirects=True)

            if response.status_code == 200:
                content = response.content

                # Validar contenido
                if self.validate_pdf_content(content):
                    st.success(f"✅ PDF downloaded successfully ({len(content)} bytes)")
                    return content
                else:
                    # Intentar decodificar respuesta para ver el error
                    try:
                        error_content = content.decode('utf-8')[:200]
                        if error_content.strip().startswith('{'):
                            error_data = json.loads(error_content)
                            st.error(f"❌ Server error: {error_data.get('message', 'PDF not available')}")
                        else:
                            st.error(f"❌ Invalid content received (it's not PDF): {error_content[:50]}...")
                    except:
                        st.error("❌ Downloaded content is not a valid PDF")
                    return None

            elif response.status_code == 404:
                st.error("❌ PDF not found (404)")
            elif response.status_code == 403:
                st.error("❌ Access denied to the PDF (403)")
            else:
                st.error(f"❌ Error downloading PDF: HTTP {response.status_code}")

        except requests.exceptions.Timeout:
            st.error("❌ Timeout descargando PDF")
        except requests.exceptions.ConnectionError:
            st.error("❌ Connection error downloading PDF")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

        return None


class ImprovedPaperProcessor(PaperProcessor):
    """Procesador mejorado con múltiples fallbacks"""

    def process_semantic_scholar_paper(self, paper: Dict, searcher: ImprovedSemanticScholarSearcher) -> Optional[Dict]:
        """Procesa un paper de Semantic Scholar con múltiples estrategias"""

        paper_id = paper.get('id', '')
        title = paper.get('title', 'Sin título')

        st.info(f"🔄 Processing paper: {title[:50]}...")

        # Estrategia 1: Intentar descargar PDF directo
        pdf_url = paper.get('pdf_url', '')
        if pdf_url:
            st.info("📥 Intentando descarga directa de PDF...")
            pdf_content = searcher.download_pdf_with_validation(pdf_url)

            if pdf_content:
                result = self.process_with_pypdf(pdf_content)
                if result:
                    result.update({
                        'title': title,
                        'abstract': paper.get('abstract', 'Not available'),
                        'authors': ', '.join(paper.get('authors', [])),
                        'year': paper.get('year', 'N/A'),
                        'source': 'Semantic Scholar PDF'
                    })
                    return result

        # Estrategia 2: Obtener más detalles del paper
        st.info("🔍 Buscando detalles adicionales del paper...")
        paper_details = searcher.get_paper_details(paper_id)

        if paper_details:
            # Intentar con URLs adicionales
            # additional_urls = []

            # Buscar en referencias externas
            if 'externalIds' in paper_details:
                external_ids = paper_details['externalIds']
                if 'DOI' in external_ids:
                    # Intentar Sci-Hub (solo mencionar la posibilidad)
                    pass
                if 'ArXiv' in external_ids:
                    arxiv_id = external_ids['ArXiv']
                    st.info(f"📄 Encontrado ID de arXiv: {arxiv_id}")
                    return self.process_with_arxiv_loader(arxiv_id)

        # Estrategia 3: Crear documento solo con metadatos disponibles
        st.warning("⚠️ No se pudo descargar PDF. Usando solo abstract y metadatos...")

        abstract = paper.get('abstract', '')
        if abstract and len(abstract) > 100:
            # Crear documento ficticio con el abstract
            from langchain.schema import Document

            metadata = {
                'title': title,
                'authors': ', '.join(paper.get('authors', [])),
                'year': paper.get('year', 'N/A'),
                'source': 'Semantic Scholar (solo abstract)',
                'venue': paper.get('venue', 'N/A')
            }

            content = f"Título: {title}\n\nAutores: {metadata['authors']}\n\nAbstract: {abstract}"

            doc = Document(page_content=content, metadata=metadata)

            return {
                'documents': [doc],
                'title': title,
                'abstract': abstract,
                'authors': metadata['authors'],
                'full_text': content,
                'source': 'Semantic Scholar (limitado)'
            }
        return None

# Función auxiliar para mejorar la interfaz
# Function modified
def process_paper_with_fallbacks(paper: Dict, source: str, searchers: Dict) -> Optional[Dict]:
    """Procesa un paper con múltiples estrategias según la fuente"""

    if source == "arXiv":
        # arXiv es más confiable
        arxiv_id = paper['id']
        result = searchers['processor'].process_with_arxiv_loader(arxiv_id)
        if result:
            st.success("✅ Paper de arXiv procesado exitosamente")
            return result
        else:
            st.error("❌ Error procesando paper de arXiv")
            return None
    elif source == "Semantic Scholar":
        # Usar el procesador mejorado
        improved_searcher = ImprovedSemanticScholarSearcher()
        improved_processor = ImprovedPaperProcessor()

        result = improved_processor.process_semantic_scholar_paper(paper, improved_searcher)

        if result:
            st.success("✅ Paper de Semantic Scholar procesado exitosamente")
            return result
        else:
            st.error("❌ No se pudo procesar el paper de Semantic Scholar")

            # Ofrecer alternativas
            st.info("💡 **Sugerencias:**")
            st.write("- Intenta con un paper diferente que tenga PDF disponible")
            st.write("- Busca el mismo paper en arXiv si es posible")
            st.write("- Algunos papers requieren acceso institucional")
            return None

    return None

# Inicializar searchers
@st.cache_resource
def init_searchers():
    semantic_api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')  # Opcional

    persistence_manager = create_persistence_manager(EMBEDDING_MODEL) # NEW

    return {
        'semantic': SemanticScholarSearcher(api_key=semantic_api_key),
        'arxiv': ArxivSearcher(),
        'processor': PaperProcessor(),
        'persistence': persistence_manager # NEW
    }

########## START NEW ##########
def process_and_persist_paper(paper: Dict, source: str, searchers: Dict) -> Optional[str]:
    """Processes paper and persist it, return collection_id if it success"""

    # Processes paper first
    processed_data = process_paper_with_fallbacks(paper, source, searchers)

    if processed_data and processed_data.get('documents'):
        paper_data = {
            'title': processed_data.get('title', paper.get('title', 'Sin título')),
            'authors': processed_data.get('authors', paper.get('authors', [])),
            'year': processed_data.get('year', paper.get('year', 'N/A')),
            'source': processed_data.get('source', source),
            'abstract': processed_data.get('abstract', paper.get('abstract', ''))
        }

        # Persists using ChromaDB
        persistence_manager = searchers['persistence']
        collection_id = persistence_manager.save_vectorstore(
            processed_data['documents'],
            paper_data
        )
        return collection_id
    return None

def load_persisted_vectorstore(collection_id: str, persistence_manager: PersistenceManager):
    """Load a persisted vectorstore"""
    return persistence_manager.load_vectorstore(collection_id)
##########  END NEW  ##########


def display_paper_card(paper: Dict, source: str, searchers: Dict = None):
    """Muestra una tarjeta de paper"""
    with st.container():
        st.markdown('<div class="paper-card">', unsafe_allow_html=True)

        # Título
        st.markdown(f"**{paper['title']}**")

        # Autores
        authors = paper.get('authors', [])
        if authors:
            authors_str = ', '.join(authors[:3])  # Mostrar solo primeros 3
            if len(authors) > 3:
                authors_str += f" y {len(authors) - 3} más"
            st.markdown(f"*Autores:* {authors_str}")

        # Año y venue/categoría
        col1, col2, col3 = st.columns(3) # Add col3 and st.columns(3)

        paper_id = paper.get('id', hash(paper['title']))
        unique_key = f"{source.lower().replace(' ', '_')}_{paper_id}"
        ########## START NEW ##########
        with col1:
            if st.button(f"📄 Process & Save", key=f"save_{unique_key}"):
                if searchers:
                    with st.spinner('Processing and saving...'):
                        # searchers = {
                        #     'semantic': st.session_state.get('semantic_searcher'),
                        #     'arxiv': st.session_state.get('arxiv_searcher'),
                        #     'processor': st.session_state.get('processor'),
                        #     'persistence': st.session_state.persistence_manager
                        # }

                        collection_id = process_and_persist_paper(paper, source, searchers)
                        if collection_id:
                            st.session_state.current_collection_id = collection_id
                            st.session_state.vectorstore = searchers['persistence'].load_vectorstore(collection_id)
                            st.success('✅ Saved paper! You can chat with it now')
                        else:
                            st.error('❌ Error processing paper')
                else:
                    st.error('❌ Searchers not available')
        ##########  END NEW  ##########
        with col2: # from col1 to col2
            year = paper.get('year', paper.get('published', 'N/A'))
            st.markdown(f"**Año:** {year}")
        with col3: # from col2 to col3
            if source == 'Semantic Scholar':
                venue = paper.get('venue', 'N/A')
                st.markdown(f"**Venue:** {venue}")
            else:  # arXiv
                category = paper.get('primary_category', 'N/A')
                st.markdown(f"**Categoría:** {category}")
        # Abstract
        abstract = paper.get('abstract', 'No disponible')
        if abstract and abstract != 'No disponible':
            with st.expander("Ver Abstract"):
                st.write(abstract[:500] + "..." if len(abstract) > 500 else abstract)
        # Métricas (solo para Semantic Scholar)
        if source == 'Semantic Scholar':
            col1, col2 = st.columns(2)
            with col1:
                citations = paper.get('citations', 0)
                st.metric("Citas", citations)
            with col2:
                influential = paper.get('influential_citations', 0)
                st.metric("Citas Influyentes", influential)
        # Etiquetas
        if source == 'arXiv':
            categories = paper.get('categories', [])
            if categories:
                st.markdown("**Categorías:**")
                for cat in categories[:3]:  # Mostrar solo primeras 3
                    st.markdown(f'<span class="subject-tag">{cat}</span>', unsafe_allow_html=True)

        # Botones de acción
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(f"📄 Procesar", key=f'process_{unique_key}'): # f"process_{paper.get('id', hash(paper['title']))}"
                st.session_state.selected_paper = paper
                st.session_state.selected_source = source
                st.rerun()

        with col2:
            url = paper.get('url', paper.get('entry_id', ''))
            if url:
                st.markdown(f"[🔗 Ver Online]({url})")

        with col3:
            pdf_url = paper.get('pdf_url', '')
            if pdf_url:
                st.markdown(f"[📥 PDF]({pdf_url})")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")


def main():
    st.title("🔬 Research Assistant")
    st.markdown("*Busca, analiza y conversa con papers académicos*")

    # Verificar configuración
    if not GOOGLE_API_KEY:
        st.error("⚠️ Google API Key no configurada. Configura GOOGLE_API_KEY en tu archivo .env")
        st.stop()

    # Inicializar searchers
    searchers = init_searchers()

    # Inicializar session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'processed_paper' not in st.session_state:
        st.session_state.processed_paper = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_collection_id' not in st.session_state: # NEW
        st.session_state.current_collection_id = None # NEW

    # Sidebar para búsqueda
    with st.sidebar:
        st.header("🔍 Search for Papers")

        # Query de búsqueda
        search_query = st.text_input(
            "Search terms:",
            placeholder="machine learning, deep learning..."
        )

        # Fuente de búsqueda
        search_source = st.selectbox(
            "Source:",
            ["Both", "Semantic Scholar", "arXiv"]
        )

        # Número de resultados
        max_results = st.slider("Resultados máximos:", 5, 50, 10)

        # Botón de búsqueda
        if st.button("🔍 Search", type="primary"):
            if search_query.strip():
                with st.spinner("Searching papers..."):
                    results = []

                    if search_source in ["Both", "Semantic Scholar"]:
                        semantic_results = searchers['semantic'].search_papers(
                            search_query, max_results
                        )
                        results.extend([(paper, "Semantic Scholar") for paper in semantic_results])

                    if search_source in ["Both", "arXiv"]:
                        arxiv_results = searchers['arxiv'].search_papers(
                            search_query, max_results
                        )
                        results.extend([(paper, "arXiv") for paper in arxiv_results])

                    st.session_state.search_results = results
                    st.success(f"Found {len(results)} papers")
            else:
                st.warning("Enter search terms")

        # Limpiar resultados
        if st.button("🗑️ Clean"):
            st.session_state.search_results = []
            st.session_state.selected_paper = None
            st.session_state.vectorstore = None
            st.session_state.processed_paper = None
            st.session_state.messages = []
            st.rerun()
        ########## START NEW ##########
        st.markdown('---')
        if st.button('📚 My Stored Papers'):
            st.session_state.show_stored_papers = True
            st.rerun()
        ##########  END NEW  ##########

    # Contenido principal
    ########## START NEW ##########
    if st.session_state.get('show_stored_papers', False):
        st.header('📚 My Stored Papers')

        if st.button('⬅️ Back to Search'):
            st.session_state.show_stored_papers = False
            st.rerun()

        persistence_manager = searchers['persistence']
        stored_papers = persistence_manager.get_all_papers()

        if stored_papers:
            for paper in stored_papers:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{paper['title']}**")
                    st.caption(f"Authors: {paper.get('authors', 'N/A')}")
                    st.caption(f"Year: {paper.get('year', 'N/A')} | Source: {paper.get('source', 'Unknown')}")
                with col2:
                    if st.button("💬", key=f"chat_{paper['collection_id']}"):
                        st.session_state.current_collection_id = paper['collection_id']
                        st.session_state.show_stored_papers = False
                        st.session_state.vectorstore = persistence_manager.load_vectorstore(paper['collection_id'])
                        st.session_state.messages = []
                        st.rerun()
                with col3:
                    if st.button("🗑️", key=f"delete_{paper['collection_id']}"):
                        persistence_manager.delete_paper(paper['collection_id'])
                        st.rerun()
                st.markdown('---')
        else:
            st.info("No papers stored")
    ##########  END NEW  ##########
    elif st.session_state.processed_paper and st.session_state.vectorstore:
        # Mostrar paper procesado y chat
        st.header("💬 Chat to the Paper")

        # Información del paper actual
        paper = st.session_state.processed_paper
        st.info(f"📄 **Current Paper:** {paper.get('title', 'Untitled')}")

        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input del usuario
        if prompt := st.chat_input("Ask a question about the paper..."):
            # Agregar mensaje del usuario
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Generar respuesta
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # Buscar documentos relevantes
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    # relevant_docs = retriever.get_relevant_documents(prompt)
                    relevant_docs = retriever.invoke(prompt)

                    # Generar respuesta
                    response = generate_answer(prompt, relevant_docs)
                    st.write(response)

                    # Agregar respuesta al historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
    elif st.session_state.selected_paper and 'selected_source' in st.session_state:
        # Procesar paper seleccionado
        paper = st.session_state.selected_paper
        source = st.session_state.selected_source

        st.header("⚙️ Processing Paper")
        st.info(f"Processing: **{paper['title']}**")

        with st.spinner("Descargando y analizando contenido..."):
            # Usar la función mejorada con fallbacks
            processed_data = process_paper_with_fallbacks(paper, source, searchers)

            if processed_data:
                # Crear vector store
                vectorstore = create_vector_store(processed_data['documents'])

                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.processed_paper = processed_data
                    st.session_state.selected_paper = None
                    st.success("✅ Paper procesado exitosamente!")
                    st.rerun()
                else:
                    st.error("Error creando vector store")
            else:
                st.error("No se pudo procesar el paper. PDF no disponible o error en descarga.")
                st.session_state.selected_paper = None
    else:
        # Mostrar resultados de búsqueda
        if st.session_state.search_results:
            st.header("📚 Resultados de Búsqueda")

            # Filtros
            col1, col2 = st.columns(2)
            with col1:
                source_filter = st.selectbox(
                    "Filtrar por fuente:",
                    ["Todas", "Semantic Scholar", "arXiv"]
                )

            # Aplicar filtros
            filtered_results = st.session_state.search_results
            if source_filter != "Todas":
                filtered_results = [
                    (paper, source) for paper, source in filtered_results
                    if source == source_filter
                ]

            st.write(f"Mostrando {len(filtered_results)} resultados")

            # Mostrar papers
            for paper, source in filtered_results:
                display_paper_card(paper, source, searchers)
        else:
            # Página de inicio
            st.markdown("""
            ## 🚀 Welcome to Research Assistant

            Esta herramienta te permite:

            - 🔍 **Buscar papers** en Semantic Scholar y arXiv
            - 📄 **Procesar PDFs** automáticamente 
            - 💬 **Hacer preguntas** específicas sobre el contenido
            - 🤖 **Obtener respuestas** usando IA avanzada

            ### Para comenzar:
            1. Usa la barra lateral para buscar papers
            2. Selecciona un paper de interés
            3. Haz clic en "Procesar" para analizarlo
            4. Comienza a hacer preguntas sobre el contenido

            ---
            *Powered by Google Gemini & Semantic Scholar API*
            """)

if __name__ == "__main__":
    main()