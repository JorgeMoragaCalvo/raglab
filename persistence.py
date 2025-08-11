import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

import streamlit as st

class PersistenceManager:
    """Gestor de persistencia para vector stores usando ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db", embedding_model=None):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Directorio para metadatos
        self.metadata_dir = self.persist_directory / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self.embedding_model = embedding_model

        # DO NOT create ChromaDB client here to avoid conflicts
        # Instead, we'll use only Langchain Chroma wrappers
        self.chroma_client = None

    def _generate_collection_id(self, paper_data: Dict) -> str:
        """Genera un ID único para la colección basado en el paper"""
        # Usar título + autores + año para crear un hash único
        title = paper_data.get('title', '')
        authors = str(paper_data.get('authors', []))
        year = str(paper_data.get('year', ''))

        # Crear hash MD5
        content = f"{title}_{authors}_{year}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitiza el nombre de la colección para ChromaDB"""
        # ChromaDB requiere nombres específicos
        sanitized = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
        # Debe empezar con letra o número
        if not sanitized[0].isalnum():
            sanitized = f"paper_{sanitized}"
        return sanitized[:63]  # Limitar longitud

    def save_vectorstore(self, documents: List[Document], paper_data: Dict) -> str:
        """Guarda un vectorstore persistente y retorna el ID de la colección"""
        try:
            # Generar ID único
            collection_id = self._generate_collection_id(paper_data)
            collection_name = f"paper_{collection_id}"

            # Verificar si ya existe
            if self.collection_exists(collection_id):
                st.warning(f"Paper ya existe en la base de datos: {paper_data.get('title', 'Sin título')}")
                return collection_id

            # Filter complex metadata from documents before creating vectorstore
            filtered_documents = filter_complex_metadata(documents)

            # Crear vectorstore con ChromaDB
            vectorstore = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embedding_model,
                collection_name=collection_name,
                persist_directory=str(self.persist_directory)
            )

            # Guardar metadatos
            metadata = {
                'collection_id': collection_id,
                'collection_name': collection_name,
                'title': paper_data.get('title', 'Sin título'),
                'authors': paper_data.get('authors', []),
                'year': paper_data.get('year', 'N/A'),
                'source': paper_data.get('source', 'Unknown'),
                'abstract': paper_data.get('abstract', '')[:500],  # Limitar tamaño
                'created_at': datetime.now().isoformat(),
                'document_count': len(documents),
                'total_chars': sum(len(doc.page_content) for doc in documents)
            }

            # Guardar metadata en archivo JSON
            metadata_file = self.metadata_dir / f"{collection_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            st.success(f"✅ Paper guardado con ID: {collection_id}")
            return collection_id

        except Exception as e:
            st.error(f"Error guardando vectorstore: {e}")
            return ""

    ########## NEW CODE ##########
    # Reset method no longer needed since we don't use ChromaDB client directly

    def load_vectorstore(self, collection_id: str):
        """Carga un vectorstore existente"""
        try:
            collection_name = f"paper_{collection_id}"

            # Verificar que existe
            if not self.collection_exists(collection_id):
                st.error(f"Colección {collection_id} no encontrada")
                return None

            # Cargar vectorstore
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=str(self.persist_directory)
            )

            return vectorstore

        except Exception as e:
            st.error(f"Error cargando vectorstore: {e}")
            return None

    def collection_exists(self, collection_id: str) -> bool:
        """Verifica si existe una colección"""
        try:
            # Check if metadata file exists (simpler and more reliable)
            metadata_file = self.metadata_dir / f"{collection_id}.json"
            return metadata_file.exists()
        except:
            return False

    def get_all_papers(self) -> List[Dict]:
        """Obtiene lista de todos los papers guardados"""
        papers = []

        try:
            # Leer todos los archivos de metadata
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    papers.append(metadata)
                except Exception as e:
                    st.warning(f"Error leyendo metadata {metadata_file.name}: {e}")

            # Ordenar por fecha de creación (más reciente primero)
            papers.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        except Exception as e:
            st.error(f"Error obteniendo lista de papers: {e}")

        return papers

    def delete_paper(self, collection_id: str) -> bool:
        """Elimina un paper de la base de datos"""
        try:
            collection_name = f"paper_{collection_id}"

            # Try to delete collection using Langchain Chroma if it exists
            try:
                if self.collection_exists(collection_id):
                    # Create a temporary Chroma instance to delete the collection
                    temp_chroma = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embedding_model,
                        persist_directory=str(self.persist_directory)
                    )
                    temp_chroma.delete_collection()
            except Exception as e:
                st.warning(f"Colección ya eliminada o no existe: {e}")

            # Eliminar archivo de metadata
            metadata_file = self.metadata_dir / f"{collection_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            st.success("✅ Paper eliminado exitosamente")
            return True

        except Exception as e:
            st.error(f"Error eliminando paper: {e}")
            return False

    def get_paper_metadata(self, collection_id: str) -> Optional[Dict]:
        """Obtiene metadatos de un paper específico"""
        try:
            metadata_file = self.metadata_dir / f"{collection_id}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error obteniendo metadata: {e}")
        return None

    def get_storage_stats(self) -> Dict:
        """Obtiene estadísticas de almacenamiento"""
        try:
            papers = self.get_all_papers()

            total_papers = len(papers)
            total_documents = sum(paper.get('document_count', 0) for paper in papers)
            total_chars = sum(paper.get('total_chars', 0) for paper in papers)

            # Calcular tamaño en disco
            total_size = 0
            for file_path in self.persist_directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            # Fuentes más comunes
            sources = [paper.get('source', 'Unknown') for paper in papers]
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1

            return {
                'total_papers': total_papers,
                'total_documents': total_documents,
                'total_characters': total_chars,
                'disk_size_mb': round(total_size / (1024 * 1024), 2),
                'source_distribution': source_counts,
                'latest_paper': papers[0].get('created_at') if papers else None
            }

        except Exception as e:
            st.error(f"Error calculando estadísticas: {e}")
            return {}

    def cleanup_orphaned_files(self):
        """Limpiar archivos huérfanos y colecciones sin metadata"""
        try:
            # Since we're not using direct ChromaDB client, we'll focus on metadata cleanup
            # This is a simplified cleanup that removes metadata files without corresponding collections
            
            st.info("🧹 Iniciando limpieza de archivos huérfanos...")
            
            # Clean up any temporary or invalid metadata files
            cleaned_count = 0
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Validate metadata structure
                    required_fields = ['collection_id', 'collection_name', 'title']
                    if not all(field in metadata for field in required_fields):
                        metadata_file.unlink()
                        st.info(f"Eliminado metadata inválido: {metadata_file.name}")
                        cleaned_count += 1
                        
                except Exception as e:
                    # Remove corrupted metadata files
                    metadata_file.unlink()
                    st.info(f"Eliminado metadata corrupto: {metadata_file.name}")
                    cleaned_count += 1

            st.success(f"✅ Limpieza completada - {cleaned_count} archivos eliminados")

        except Exception as e:
            st.error(f"Error en limpieza: {e}")

def create_persistence_manager(embedding_model) -> PersistenceManager:
    """Factory function para crear un PersistenceManager"""
    return PersistenceManager(embedding_model=embedding_model)