"""
Módulo Analyzer Principal
Orquesta todo el análisis de InterOrdra
"""

from backend.embeddings import SemanticEmbedder
from backend.clustering import ConceptClusterer
from backend.gap_detector import GapDetector
import numpy as np
from typing import Dict

class InterOrdraAnalyzer:
    """
    Analizador principal de InterOrdra.
    
    Coordina:
    - Generación de embeddings
    - Clustering de conceptos
    - Detección de gaps
    - Visualización de datos
    """
    
    def __init__(self, language: str = 'es', threshold: float = 0.5):
        """
        Inicializa el analyzer.
        
        Args:
            language: 'es' o 'en'
            threshold: Umbral de similaridad para considerar acoplamiento
        """
        print("Inicializando InterOrdra...")
        
        self.embedder = SemanticEmbedder(language=language)
        self.clusterer = ConceptClusterer(eps=0.5, min_samples=2)
        self.gap_detector = GapDetector(threshold=threshold, language=language)
        
        print("InterOrdra listo para analizar\n")
    
    def analyze(self, text_a: str, text_b: str, label_a: str = "Texto A", label_b: str = "Texto B") -> Dict:
        """
        Pipeline completo de análisis.
        
        Args:
            text_a: Primer texto (ej: documentación técnica)
            text_b: Segundo texto (ej: pregunta de usuario)
            label_a: Etiqueta descriptiva para texto A
            label_b: Etiqueta descriptiva para texto B
        
        Returns:
            Dict con todos los resultados del análisis
        """
        print(f"Analizando: '{label_a}' vs '{label_b}'")
        print("-" * 60)
        
        # 1. GENERAR EMBEDDINGS
        print("[1/4] Generando embeddings semanticos...")
        data_a = self.embedder.embed_text(text_a)
        data_b = self.embedder.embed_text(text_b)
        print(f"   *{label_a}: {data_a['num_sentences']} oraciones")
        print(f"   *{label_b}: {data_b['num_sentences']} oraciones")
        
        # 2. DETECTAR CLUSTERS
        print("\n[2/4] Identificando clusters conceptuales...")
        clusters_a = self.clusterer.find_clusters(
            data_a['embeddings'], 
            data_a['sentences']
        )
        clusters_b = self.clusterer.find_clusters(
            data_b['embeddings'], 
            data_b['sentences']
        )
        print(f"   *{label_a}: {len([c for c in clusters_a if c != -1])} clusters")
        print(f"   *{label_b}: {len([c for c in clusters_b if c != -1])} clusters")
        
        # 3. DETECTAR GAPS
        print("\n[3/4] Detectando desacoplamientos...")
        gaps = self.gap_detector.detect_gaps(data_a, data_b)
        print(f"   *Similaridad global: {gaps['global_similarity']:.1%}")
        print(f"   * Estado: {'ACOPLADO' if gaps['is_coupled'] else 'DESACOPLADO'}")
        
        # 4. PREPARAR VISUALIZACIÓN
        print("\n[4/4] Preparando datos de visualizacion...")
        all_embeddings = np.vstack([
            data_a['embeddings'],
            data_b['embeddings']
        ])
        coords_3d = self.clusterer.reduce_dimensions(all_embeddings, n_components=3)
        
        # Separar coordenadas
        split_point = len(data_a['sentences'])
        coords_a = coords_3d[:split_point]
        coords_b = coords_3d[split_point:]
        
        # 5. COMPILAR RESULTADOS
        results = {
            'metadata': {
                'label_a': label_a,
                'label_b': label_b,
                'threshold': self.gap_detector.threshold
            },
            'text_a': {
                'sentences': data_a['sentences'],
                'num_sentences': data_a['num_sentences'],
                'clusters': clusters_a,
                'num_clusters': len([c for c in clusters_a if c != -1]),
                'coords_3d': coords_a.tolist()
            },
            'text_b': {
                'sentences': data_b['sentences'],
                'num_sentences': data_b['num_sentences'],
                'clusters': clusters_b,
                'num_clusters': len([c for c in clusters_b if c != -1]),
                'coords_3d': coords_b.tolist()
            },
            'gaps': gaps,
            'summary': self.gap_detector.generate_summary(gaps)
        }
        
        print("\nAnalisis completado\n")
        
        return results
    
    def quick_analysis(self, text_a: str, text_b: str) -> None:
        """
        Análisis rápido con output formateado a consola.
        
        Args:
            text_a, text_b: Textos a comparar
        """
        results = self.analyze(text_a, text_b)
        
        # Mostrar resumen
        print(results['summary'])
        
        # Mostrar algunos gaps específicos
        if results['gaps']['gaps']['text_a_orphans']:
            print("Ejemplos de conceptos huerfanos en Texto A:")
            for orphan in results['gaps']['gaps']['text_a_orphans'][:2]:
                print(f"   *\"{orphan['sentence']}\"")
                print(f"     (mejor match: {orphan['best_match_similarity']:.1%})")
        
        if results['gaps']['gaps']['text_b_orphans']:
            print("\nEjemplos de conceptos huerfanos en Texto B:")
            for orphan in results['gaps']['gaps']['text_b_orphans'][:2]:
                print(f"   *\"{orphan['sentence']}\"")
                print(f"     (mejor match: {orphan['best_match_similarity']:.1%})")
        
        # Mostrar vocabulario único
        vocab = results['gaps']['vocabulary_analysis']
        if vocab['unique_to_a']:
            print(f"\nVocabulario unico en A: {', '.join(vocab['unique_to_a'][:5])}")
        if vocab['unique_to_b']:
            print(f"Vocabulario unico en B: {', '.join(vocab['unique_to_b'][:5])}")


def test_analyzer():
    """Test completo de InterOrdra"""
    print("\n" + "="*60)
    print("PROBANDO INTERORDRA - ANALISIS COMPLETO")
    print("="*60 + "\n")
    
    # Crear analyzer
    analyzer = InterOrdraAnalyzer(language='es', threshold=0.5)
    
    # Textos de ejemplo: documentación técnica vs pregunta de usuario
    texto_tecnico = """
    El sistema utiliza embeddings vectoriales de alta dimensión para representar 
    información semántica. Los transformers pre-entrenados generan representaciones 
    contextuales mediante mecanismos de atención multi-cabeza. La arquitectura 
    implementa capas de normalización y conexiones residuales para estabilidad.
    """
    
    pregunta_usuario = """
    ¿Cómo funciona esto? No entiendo qué significa que use vectores.
    ¿Qué es un transformer? ¿Por qué necesito saber de matemáticas avanzadas?
    Solo quiero que mi aplicación entienda texto en español.
    """
    
    # Analizar
    analyzer.quick_analysis(
        texto_tecnico, 
        pregunta_usuario
    )
    
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_analyzer()