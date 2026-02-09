"""
Módulo de Embeddings Semánticos
Convierte texto en representaciones vectoriales
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List
from backend.simple_splitter import split_sentences

class SemanticEmbedder:
    """
    Genera embeddings semánticos de textos usando sentence-transformers.
    
    Los embeddings capturan significado en vectores de alta dimensión,
    permitiendo comparaciones matemáticas de similaridad semántica.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', language: str = 'es'):
        """
        Inicializa el embedder con un modelo pre-entrenado.
        
        Args:
            model_name: Nombre del modelo de sentence-transformers
                - 'all-MiniLM-L6-v2': Multilingüe, rápido, 384 dimensiones
                - 'paraphrase-multilingual-mpnet-base-v2': Más preciso, 768 dim
            language: 'es' o 'en' para procesamiento de oraciones
        """
        print(f"Cargando modelo {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.language = language
        
        print(f"Modelo cargado. Dimension de embeddings: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str, return_global: bool = True) -> Dict:
        """
        Convierte texto en embeddings vectoriales.
        
        Args:
            text: Texto a procesar
            return_global: Si True, también retorna embedding global del texto completo
        
        Returns:
            Dict con:
                - 'sentences': Lista de oraciones
                - 'embeddings': Array numpy con vectores (shape: [n_sentences, embedding_dim])
                - 'global_embedding': Vector del texto completo (si return_global=True)
                - 'num_sentences': Número de oraciones detectadas
        """
        # Dividir en oraciones
        sentences = split_sentences(text, self.language)
        
        if not sentences:
            raise ValueError("No se detectaron oraciones en el texto")
        
        # Generar embeddings para cada oración
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        
        result = {
            'sentences': sentences,
            'embeddings': embeddings,
            'num_sentences': len(sentences)
        }
        
        # Embedding global (promedio o del texto completo)
        if return_global:
            global_embedding = self.model.encode([text], convert_to_numpy=True)[0]
            result['global_embedding'] = global_embedding
        
        return result
    
    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Calcula similaridad coseno entre dos embeddings.
        
        Args:
            embedding_a, embedding_b: Vectores a comparar
        
        Returns:
            Similaridad entre 0 (nada similar) y 1 (idéntico)
        """
        # Normalizar vectores
        norm_a = embedding_a / np.linalg.norm(embedding_a)
        norm_b = embedding_b / np.linalg.norm(embedding_b)
        
        # Producto punto = similaridad coseno
        similarity = np.dot(norm_a, norm_b)
        
        return float(similarity)


# Función de testing
def test_embedder():
    """Prueba rápida del embedder"""
    print("\nProbando SemanticEmbedder...\n")
    
    # Crear embedder
    embedder = SemanticEmbedder(language='es')
    
    # Texto de ejemplo
    texto = "El lenguaje es una vibración. Los sistemas complejos se comunican mediante patrones."
    
    # Generar embeddings
    result = embedder.embed_text(texto)
    
    # Mostrar resultados
    print("Resultados:")
    print(f"Oraciones detectadas: {result['num_sentences']}")
    print(f"Dimension de embeddings: {result['embeddings'].shape[1]}")
    print("Oraciones:")
    for i, sent in enumerate(result['sentences'], 1):
        print(f"  {i}. {sent}")
    
    # Calcular similaridad entre oraciones
    if result['num_sentences'] >= 2:
        sim = embedder.compute_similarity(result['embeddings'][0], result['embeddings'][1])
        print(f"Similaridad entre oracion 1 y 2: {sim:.3f}")
    
    print("\nTest completado exitosamente!")


if __name__ == "__main__":
    test_embedder()