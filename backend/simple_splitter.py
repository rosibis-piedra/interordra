"""
Simple sentence splitter sin dependencias pesadas
"""

import re

def split_sentences(text: str, language: str = 'es') -> list:
    """
    Divide texto en oraciones usando regex simple.
    
    Args:
        text: Texto a dividir
        language: 'es' o 'en' (no afecta el splitting por ahora)
    
    Returns:
        Lista de oraciones
    """
    # Dividir por puntos, signos de exclamación, interrogación
    sentences = re.split(r'[.!?]+', text)
    
    # Limpiar y filtrar vacías
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def extract_keywords(text: str, language: str = 'es') -> set:
    """
    Extrae palabras clave simples sin spaCy.
    
    Args:
        text: Texto a analizar
        language: 'es' o 'en'
    
    Returns:
        Set de keywords
    """
    # Stopwords básicas
    stopwords_es = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber', 
                    'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
                    'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
                    'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy', 'sin', 'vez',
                    'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo', 'yo', 'también',
                    'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero', 'desde', 'grande',
                    'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí', 'día', 'uno',
                    'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa', 'tanto', 'hombre', 'parecer',
                    'nuestro', 'tan', 'donde', 'ahora', 'parte', 'después', 'vida', 'quedar', 'siempre',
                    'creer', 'hablar', 'llevar', 'dejar', 'nada', 'cada', 'seguir', 'menos', 'nuevo',
                    'encontrar', 'algo', 'solo', 'decir', 'puede', 'sido', 'esta', 'son', 'tiene',
                    'han', 'fue', 'era', 'esa', 'estos', 'estas', 'del', 'los', 'las', 'una', 'unos', 'unas'}
    
    stopwords_en = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
                    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
                    'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
                    'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
                    'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
                    'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
                    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
                    'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
                    'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is',
                    'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did', 'having', 'may', 'should'}
    
    stopwords = stopwords_es if language == 'es' else stopwords_en
    
    # Tokenizar simple
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filtrar stopwords y palabras cortas
    keywords = {w for w in words if w not in stopwords and len(w) > 3}
    
    return keywords