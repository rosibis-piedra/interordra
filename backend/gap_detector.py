"""
Módulo de Detección de Gaps Semánticos
Identifica desacoplamientos entre dos textos
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from typing import Dict, List, Tuple

class GapDetector:
    """
    Detecta brechas semánticas entre dos textos.
    
    Identifica:
    - Distancia semántica global
    - Conceptos "huérfanos" (sin equivalente en el otro texto)
    - Vocabulario único vs compartido
    - Severidad del desacoplamiento
    """
    
    def __init__(self, threshold: float = 0.5, language: str = 'es'):
        """
        Inicializa el detector de gaps.
        
        Args:
            threshold: Similaridad mínima para considerar "acoplado"
                - 0.7+: Muy similar (bien acoplado)
                - 0.5-0.7: Moderadamente similar
                - <0.5: Poco similar (desacoplado)
            language: 'es' o 'en' para análisis de vocabulario
        """
        self.threshold = threshold
        self.language = language
        
        # Cargar spaCy para análisis de vocabulario
        if language == 'es':
    try:
        self.nlp = spacy.load('es_core_news_sm')
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
        self.nlp = spacy.load('es_core_news_sm')
else:
    try:
        self.nlp = spacy.load('en_core_web_sm')
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        self.nlp = spacy.load('en_core_web_sm')
        """
        Encuentra desacoplamientos entre dos textos.
        
        Args:
            text_a_data: Output de SemanticEmbedder.embed_text() para texto A
            text_b_data: Output de SemanticEmbedder.embed_text() para texto B
        
        Returns:
            Dict con análisis completo de gaps:
                - global_similarity: Similaridad global (0-1)
                - is_coupled: Bool indicando si están acoplados
                - gaps: Info detallada de desacoplamientos
                - vocabulary_analysis: Análisis de vocabulario
        """
        results = {}
        
        # 1. SIMILARIDAD GLOBAL
        global_sim = cosine_similarity(
            [text_a_data['global_embedding']], 
            [text_b_data['global_embedding']]
        )[0][0]
        
        results['global_similarity'] = float(global_sim)
        results['is_coupled'] = global_sim > self.threshold
        
        # 2. ANÁLISIS SENTENCE-BY-SENTENCE
        sentence_similarities = cosine_similarity(
            text_a_data['embeddings'],
            text_b_data['embeddings']
        )
        
        # Oraciones de A sin match cercano en B
        max_similarities_a = np.max(sentence_similarities, axis=1)
        orphan_sentences_a = [
            {
                'sentence': text_a_data['sentences'][i],
                'max_similarity': float(max_similarities_a[i]),
                'best_match_in_b': text_b_data['sentences'][np.argmax(sentence_similarities[i])],
                'best_match_similarity': float(np.max(sentence_similarities[i]))
            }
            for i in range(len(text_a_data['sentences']))
            if max_similarities_a[i] < self.threshold
        ]
        
        # Oraciones de B sin match cercano en A
        max_similarities_b = np.max(sentence_similarities, axis=0)
        orphan_sentences_b = [
            {
                'sentence': text_b_data['sentences'][i],
                'max_similarity': float(max_similarities_b[i]),
                'best_match_in_a': text_a_data['sentences'][np.argmax(sentence_similarities[:, i])],
                'best_match_similarity': float(np.max(sentence_similarities[:, i]))
            }
            for i in range(len(text_b_data['sentences']))
            if max_similarities_b[i] < self.threshold
        ]
        
        results['gaps'] = {
            'text_a_orphans': orphan_sentences_a,
            'text_b_orphans': orphan_sentences_b,
            'gap_severity': float(1 - global_sim),
            'orphan_count_a': len(orphan_sentences_a),
            'orphan_count_b': len(orphan_sentences_b)
        }
        
        # 3. ANÁLISIS DE VOCABULARIO
        results['vocabulary_analysis'] = self._analyze_vocabulary(
            text_a_data, text_b_data
        )
        
        # 4. GENERAR RECOMENDACIONES
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _analyze_vocabulary(self, text_a_data: Dict, text_b_data: Dict) -> Dict:
        """
        Detecta palabras técnicas/únicas en cada texto.
        
        Args:
            text_a_data, text_b_data: Datos de embeddings
        
        Returns:
            Dict con análisis de vocabulario
        """
        def extract_keywords(text: str) -> set:
            """Extrae palabras clave importantes"""
            doc = self.nlp(text)
            return {
                token.lemma_.lower() 
                for token in doc 
                if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                and not token.is_stop
                and len(token.text) > 3
            }
        
        # Extraer keywords de ambos textos
        text_a_full = ' '.join(text_a_data['sentences'])
        text_b_full = ' '.join(text_b_data['sentences'])
        
        keywords_a = extract_keywords(text_a_full)
        keywords_b = extract_keywords(text_b_full)
        
        # Calcular overlap
        shared = keywords_a & keywords_b
        unique_a = keywords_a - keywords_b
        unique_b = keywords_b - keywords_a
        
        total_unique = len(keywords_a | keywords_b)
        overlap_ratio = len(shared) / total_unique if total_unique > 0 else 0
        
        return {
            'unique_to_a': sorted(list(unique_a))[:20],  # Top 20
            'unique_to_b': sorted(list(unique_b))[:20],
            'shared': sorted(list(shared))[:20],
            'vocabulary_overlap': float(overlap_ratio),
            'total_keywords_a': len(keywords_a),
            'total_keywords_b': len(keywords_b)
        }
    
    def _generate_recommendations(self, results: Dict) -> List[Dict]:
        """
        Genera recomendaciones para cerrar los gaps detectados.
        
        Args:
            results: Resultados del análisis
        
        Returns:
            Lista de recomendaciones con severidad
        """
        recommendations = []
        
        sim = results['global_similarity']
        orphans_a = results['gaps']['orphan_count_a']
        orphans_b = results['gaps']['orphan_count_b']
        vocab_overlap = results['vocabulary_analysis']['vocabulary_overlap']
        
        # Recomendación según similaridad global
        if sim < 0.3:
            recommendations.append({
                'severity': 'high',
                'category': 'global_coupling',
                'message': 'Desacoplamiento severo: los textos tratan temas muy diferentes',
                'suggestion': 'Considerar si están dirigidos a la misma audiencia o propósito. Puede ser necesario reestructurar completamente uno de los textos.'
            })
        elif sim < 0.5:
            recommendations.append({
                'severity': 'medium',
                'category': 'global_coupling',
                'message': 'Desacoplamiento moderado: hay overlap pero también brechas significativas',
                'suggestion': 'Identificar conceptos clave compartidos y expandir desde ahí. Agregar puentes conceptuales explícitos.'
            })
        else:
            recommendations.append({
                'severity': 'low',
                'category': 'global_coupling',
                'message': 'Acoplamiento aceptable: los textos están razonablemente alineados',
                'suggestion': 'Enfocarse en refinar detalles específicos mencionados en gaps.'
            })
        
        # Recomendación según conceptos huérfanos
        if orphans_a > 0:
            recommendations.append({
                'severity': 'info',
                'category': 'orphan_concepts_a',
                'message': f'Texto A contiene {orphans_a} concepto(s) sin equivalente cercano en Texto B',
                'suggestion': 'Considerar agregar estos conceptos a Texto B o verificar si son realmente necesarios.',
                'details': results['gaps']['text_a_orphans'][:3]
            })
        
        if orphans_b > 0:
            recommendations.append({
                'severity': 'info',
                'category': 'orphan_concepts_b',
                'message': f'Texto B contiene {orphans_b} concepto(s) sin equivalente cercano en Texto A',
                'suggestion': 'Considerar agregar estos conceptos a Texto A o verificar si son realmente necesarios.',
                'details': results['gaps']['text_b_orphans'][:3]
            })
        
        # Recomendación según vocabulario
        if vocab_overlap < 0.3:
            recommendations.append({
                'severity': 'medium',
                'category': 'vocabulary',
                'message': f'Vocabulario compartido muy bajo ({vocab_overlap:.1%})',
                'suggestion': 'Los textos usan lenguajes muy diferentes. Considerar crear un glosario de términos equivalentes o usar vocabulario más uniforme.'
            })
        
        return recommendations
    
    def generate_summary(self, results: Dict) -> str:
        """
        Genera resumen legible del análisis.
        
        Args:
            results: Output de detect_gaps()
        
        Returns:
            String con resumen formateado
        """
        lines = ["\n" + "="*60]
        lines.append("📊 ANÁLISIS DE DESACOPLAMIENTO SEMÁNTICO")
        lines.append("="*60)
        
        # Métricas globales
        lines.append(f"\n🔗 Similaridad Global: {results['global_similarity']:.1%}")
        lines.append(f"{'✅ ACOPLADO' if results['is_coupled'] else '❌ DESACOPLADO'}")
        lines.append(f"⚠️  Severidad del Gap: {results['gaps']['gap_severity']:.1%}")
        
        # Vocabulario
        vocab = results['vocabulary_analysis']
        lines.append(f"\n📚 Vocabulario:")
        lines.append(f"  • Overlap: {vocab['vocabulary_overlap']:.1%}")
        lines.append(f"  • Único en A: {len(vocab['unique_to_a'])} palabras")
        lines.append(f"  • Único en B: {len(vocab['unique_to_b'])} palabras")
        
        # Conceptos huérfanos
        lines.append(f"\n🔸 Conceptos Huérfanos:")
        lines.append(f"  • En Texto A: {results['gaps']['orphan_count_a']}")
        lines.append(f"  • En Texto B: {results['gaps']['orphan_count_b']}")
        
        # Recomendaciones
        lines.append(f"\n💡 Recomendaciones:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢', 'info': 'ℹ️'}
            emoji = severity_emoji.get(rec['severity'], '•')
            lines.append(f"  {emoji} {rec['message']}")
        
        lines.append("="*60 + "\n")
        
        return "\n".join(lines)


# NO incluir test aquí para evitar el problema de torch
# El test se hará desde analyzer.py

if __name__ == "__main__":
    print("⚠️  Ejecutá el test desde analyzer.py para evitar problemas de imports")