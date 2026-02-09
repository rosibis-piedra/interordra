"""
Módulo de Clustering Conceptual
Agrupa oraciones semánticamente similares en clusters
"""

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
from typing import Dict, List, Optional

class ConceptClusterer:
    """
    Identifica clusters de conceptos relacionados en textos.
    
    Usa DBSCAN (Density-Based Spatial Clustering) que:
    - No requiere especificar número de clusters anticipadamente
    - Detecta clusters de forma arbitraria
    - Identifica "ruido" (conceptos aislados)
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Inicializa el clusterer.
        
        Args:
            eps: Distancia máxima para considerar dos puntos vecinos
                - Menor eps = clusters más estrictos/pequeños
                - Mayor eps = clusters más permisivos/grandes
            min_samples: Mínimo de puntos para formar un cluster
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def find_clusters(self, embeddings: np.ndarray, sentences: List[str]) -> Dict:
        """
        Agrupa oraciones semánticamente similares.
        
        Args:
            embeddings: Array numpy con vectores semánticos (shape: [n_sentences, embed_dim])
            sentences: Lista de oraciones originales
        
        Returns:
            Dict con información de clusters:
                {
                    cluster_id: {
                        'sentences': [...],
                        'centroid': vector_promedio,
                        'size': número_de_oraciones
                    }
                }
        """
        if len(embeddings) == 0:
            return {}
        
        # Aplicar DBSCAN con distancia coseno
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine'
        ).fit(embeddings)
        
        labels = clustering.labels_
        
        # Organizar resultados por cluster
        clusters = {}
        noise_items = []
        
        for idx, label in enumerate(labels):
            # Label -1 = ruido (no pertenece a ningún cluster)
            if label == -1:
                noise_items.append({
                    'sentence': sentences[idx],
                    'embedding': embeddings[idx]
                })
                continue
            
            # Crear cluster si no existe
            if label not in clusters:
                clusters[label] = []
            
            clusters[label].append({
                'sentence': sentences[idx],
                'embedding': embeddings[idx]
            })
        
        # Calcular centroides y metadata para cada cluster
        processed_clusters = {}
        for label, items in clusters.items():
            # Centroide = promedio de todos los embeddings del cluster
            embeddings_array = np.array([item['embedding'] for item in items])
            centroid = np.mean(embeddings_array, axis=0)
            
            processed_clusters[label] = {
                'sentences': [item['sentence'] for item in items],
                'centroid': centroid,
                'size': len(items),
                'density': self._calculate_density(embeddings_array)
            }
        
        # Agregar items de ruido como pseudo-cluster
        if noise_items:
            processed_clusters[-1] = {
                'sentences': [item['sentence'] for item in noise_items],
                'centroid': None,
                'size': len(noise_items),
                'density': 0.0,
                'is_noise': True
            }
        
        return processed_clusters
    
    def _calculate_density(self, embeddings: np.ndarray) -> float:
        """
        Calcula densidad del cluster (qué tan compactos están los puntos).
        
        Args:
            embeddings: Vectores del cluster
        
        Returns:
            Densidad promedio (0 = muy disperso, 1 = muy compacto)
        """
        if len(embeddings) <= 1:
            return 1.0
        
        # Calcular distancia promedio al centroide
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalizar: menor distancia = mayor densidad
        density = 1.0 / (1.0 + avg_distance)
        
        return float(density)
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
        """
        Reduce dimensionalidad para visualización.
        
        Embeddings típicamente tienen 384 o 768 dimensiones.
        Los reducimos a 2D o 3D para graficar.
        
        Args:
            embeddings: Vectores de alta dimensión
            n_components: Dimensiones finales (2 o 3)
        
        Returns:
            Array con vectores reducidos
        """
        if len(embeddings) < n_components:
            # Si hay menos puntos que componentes, usar PCA con menos componentes
            n_components = min(n_components, len(embeddings))
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        
        return reduced
    
    def get_cluster_summary(self, clusters: Dict) -> str:
        """
        Genera resumen legible de los clusters encontrados.
        
        Args:
            clusters: Output de find_clusters()
        
        Returns:
            String con resumen
        """
        summary_lines = ["\nResumen de Clusters:\n"]
        
        for cluster_id, cluster_info in sorted(clusters.items()):
            if cluster_id == -1:
                summary_lines.append(f"Conceptos aislados ({cluster_info['size']}):")
            else:
                summary_lines.append(
                    f"Cluster {cluster_id} ({cluster_info['size']} oraciones, "
                    f"densidad: {cluster_info['density']:.2f}):"
                )
            
            # Mostrar primeras 3 oraciones del cluster
            for i, sentence in enumerate(cluster_info['sentences'][:3]):
                summary_lines.append(f"   {i+1}. {sentence}")
            
            if cluster_info['size'] > 3:
                summary_lines.append(f"   ... y {cluster_info['size'] - 3} más")
            
            summary_lines.append("")
        
        return "\n".join(summary_lines)


# Función de testing
def test_clusterer():
    """Prueba rápida del clusterer"""
    from embeddings import SemanticEmbedder
    
    print("\nProbando ConceptClusterer...\n")
    
    # Crear embedder
    embedder = SemanticEmbedder(language='es')
    
    # Texto de ejemplo con varios conceptos
    texto = """
    El lenguaje es una vibración entre sistemas.
    Los sistemas complejos se comunican mediante patrones.
    La inteligencia artificial procesa embeddings vectoriales.
    Las redes neuronales aprenden de datos.
    La música es una forma de comunicación no verbal.
    El canto transmite emociones profundas.
    """
    
    # Generar embeddings
    result = embedder.embed_text(texto)
    
    # Crear clusterer y buscar grupos
    clusterer = ConceptClusterer(eps=0.4, min_samples=2)
    clusters = clusterer.find_clusters(result['embeddings'], result['sentences'])
    
    # Mostrar resumen
    print(clusterer.get_cluster_summary(clusters))
    
    print("Test completado!")


if __name__ == "__main__":
    test_clusterer()