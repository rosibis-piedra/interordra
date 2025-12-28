"""
Módulo de Embeddings Semánticos
Convierte texto en representaciones vectoriales
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from typing import Dict, List

class SemanticEmbedder:
    """
    Convierte texto en vectores semánticos usando transformers pre-entrenados.
    
    Los embeddings capturan significado: palabras/frases similares
    tendrán vectores cercanos en el espacio de alta dimensión.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', language: str = 'es'):
        """
        Inicializa el embedder.
        
        Args:
            model_name: Modelo de sentence-transformers a usar
                - 'all-MiniLM-L6-v2': Rápido, inglés/español, 384 dims
                - 'paraphrase-multilingual-mpnet-base-v2': Más preciso, multilingüe, 768 dims
            language: 'es' o 'en' para procesamiento de oraciones
        """
        print(f"🔄 Cargando modelo {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.language = language
        
        # Cargar modelo de spaCy para división de oraciones
        # Cargar modelo de spaCy para división de oraciones
        
        if language == 'es':
            try:
                self.nlp = spacy.load('es-core-news-sm')
            except:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
                self.nlp = spacy.load('es-core-news-sm')
        else:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load('en_core_web_sm')
        
        print(f"✅ Modelo cargado. Dimensión de embeddings: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str, split_sentences: bool = True) -> Dict:
        """
        Convierte texto en embeddings vectoriales.
        
        Args:
            text: Texto de entrada
            split_sentences: Si True, genera embedding por oración individual
        
        Returns:
            Dict con:
                - 'sentences': Lista de oraciones (si split_sentences=True)
                - 'embeddings': Array numpy con vectores (shape: [n_sentences, embed_dim])
                - 'global_embedding': Vector promedio del texto completo
                - 'text': Texto original (si split_sentences=False)
        """
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")
        
        if split_sentences:
            # Dividir en oraciones
            sentences = self._split_sentences(text)
            
            if not sentences:
                raise ValueError("No se pudieron extraer oraciones del texto")
            
            # Generar embeddings para cada oración
            embeddings = self.model.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Embedding global = promedio de todas las oraciones
            global_embedding = np.mean(embeddings, axis=0)
            
            return {
                'sentences': sentences,
                'embeddings': embeddings,
                'global_embedding': global_embedding,
                'num_sentences': len(sentences),
                'embedding_dim': embeddings.shape[1]
            }
        else:
            # Generar embedding del texto completo
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            
            return {
                'text': text,
                'embedding': embedding,
                'embedding_dim': len(embedding)
            }
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Divide texto en oraciones usando spaCy.
        
        Args:
            text: Texto a dividir
        
        Returns:
            Lista de oraciones
        """
        # Procesar con spaCy
        doc = self.nlp(text)
        
        # Extraer oraciones, limpiando espacios
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        return sentences
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula similaridad coseno entre dos embeddings.
        
        Similaridad coseno mide el ángulo entre vectores:
        - 1.0 = idénticos
        - 0.0 = ortogonales (no relacionados)
        - -1.0 = opuestos
        
        Args:
            embedding1, embedding2: Vectores a comparar
        
        Returns:
            Similaridad en rango [0, 1]
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape si es necesario
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        return float(similarity)


# Función de utilidad para testing rápido
def test_embedder():
    """Prueba rápida del embedder"""
    print("\n🧪 Probando SemanticEmbedder...\n")
    
    embedder = SemanticEmbedder(language='es')
    
    texto = "El lenguaje es una vibración. Los sistemas complejos se comunican mediante patrones."
    
    result = embedder.embed_text(texto)
    
    print(f"\n📊 Resultados:")
    print(f"Oraciones detectadas: {result['num_sentences']}")
    print(f"Dimensión de embeddings: {result['embedding_dim']}")
    print(f"\nOraciones:")
    for i, sent in enumerate(result['sentences']):
        print(f"  {i+1}. {sent}")
    
    # Probar similaridad entre oraciones
    if result['num_sentences'] >= 2:
        sim = embedder.compute_similarity(
            result['embeddings'][0],
            result['embeddings'][1]
        )
        print(f"\nSimilaridad entre oración 1 y 2: {sim:.3f}")
    
    print("\n✅ Test completado exitosamente!")


if __name__ == "__main__":
    # Ejecutar test si se corre directamente
    test_embedder()