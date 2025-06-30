"""
Legal RAG QA System (CLI Only)
Author: Shivani Ale
Description: Command-line tool for answering legal questions using retrieval-augmented generation (RAG) over a legal passage dataset.
"""

import os
import json
import time
import click
import pandas as pd
import numpy as np
import sentence_transformers
import faiss
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

class LegalRAGQA:
    def __init__(self, csv_path="top_10000_data.csv", json_path="passage_dict.json"):
        self.csv_path = csv_path
        self.json_path = json_path
        self.df = None
        self.passage_dict = None
        self.embeddings = None
        self.index = None
        self.model = None
        self.passage_texts = []
        self.passage_ids = []

    def load_data(self):
        """Load CSV and JSON data into memory."""
        self.df = pd.read_csv(self.csv_path)
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            self.passage_dict = data['data']
        self.passage_texts = []
        self.passage_ids = []
        for pid in self.df['passage_id'].unique():
            if pid in self.passage_dict:
                self.passage_texts.append(self.passage_dict[pid])
                self.passage_ids.append(pid)

    def create_embeddings(self):
        """Create embeddings for all passages and build a FAISS index."""
        self.model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.passage_texts, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype('float32'))

    def retrieve_passages(self, query, top_k=5):
        """Retrieve the most relevant passages for a given query."""
        query_emb = self.model.encode([query])
        scores, idxs = self.index.search(query_emb.astype('float32'), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            pid = self.passage_ids[idx]
            text = self.passage_texts[idx]
            row = self.df[self.df['passage_id'] == pid].iloc[0]
            results.append({
                'passage_id': pid,
                'text': text,
                'score': float(score),
                'source_cite': row['source_cite'],
                'source_name': row['source_name'],
                'source_court': row['source_court'],
                'source_date': row['source_date'],
                'quote': row['quote'],
                'destination_context': row['destination_context']
            })
        return results

    def generate_answer(self, question, passages):
        """Generate an answer using OpenAI API based on retrieved passages."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "[ERROR] OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
        try:
            client = openai.OpenAI(api_key=api_key)
            context = "\n\n".join([
                f"Passage {i+1} (from {p['source_cite']}):\n{p['text']}"
                for i, p in enumerate(passages)
            ])
            prompt = (
                "You are a legal assistant. Answer the following question using only the provided legal passages. "
                "Cite sources when possible.\n\n"
                f"Question: {question}\n\nRelevant Legal Passages:\n{context}\n\nAnswer:"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a legal assistant. Provide clear, well-reasoned answers based on the provided legal passages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[ERROR] Could not generate answer: {str(e)}"

    def answer_question(self, question, top_k=5):
        """Run the full RAG pipeline: retrieve passages and generate an answer."""
        start_retrieval = time.time()
        passages = self.retrieve_passages(question, top_k)
        retrieval_time = time.time() - start_retrieval
        start_gen = time.time()
        answer = self.generate_answer(question, passages)
        gen_time = time.time() - start_gen
        return {
            'question': question,
            'answer': answer,
            'passages': passages,
            'retrieval_time': retrieval_time,
            'generation_time': gen_time
        }

@click.group()
def cli():
    """Legal RAG QA System: Ask legal questions and get answers from real case law."""
    pass

@cli.command()
@click.option('--query', '-q', help='Your legal question (in quotes)')
@click.option('--top-k', '-k', default=5, help='How many relevant passages to retrieve (default: 5)')
def ask(query, top_k):
    """Ask a legal question and get an answer with supporting passages."""
    print("\n[INFO] Loading data and building search index...")
    rag = LegalRAGQA()
    rag.load_data()
    rag.create_embeddings()
    print("[INFO] Ready! Processing your question...")
    result = rag.answer_question(query, top_k)
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nTop {top_k} Relevant Passages:")
    for i, p in enumerate(result['passages'], 1):
        print(f"\n{i}. {p['source_cite']} (Score: {p['score']:.3f})")
        print(f"   {p['text'][:200]}...")
    print(f"\n[INFO] Retrieval time: {result['retrieval_time']:.2f}s | Generation time: {result['generation_time']:.2f}s")

@cli.command()
@click.option('--top-k', '-k', default=5, help='How many relevant passages to retrieve (default: 5)')
def synthetic(top_k):
    """Run a batch of example legal questions and see performance."""
    example_questions = [
        "What is a violent crime as it pertains to career offender status?",
        "How do courts define 'violent felony' under the ACCA?",
        "What constitutes a crime of violence for sentencing purposes?",
        "How do courts interpret the 'residual clause' of the ACCA?",
        "What is a common law marriage?"
    ]
    print("\n[INFO] Loading data and building search index...")
    rag = LegalRAGQA()
    rag.load_data()
    rag.create_embeddings()
    print("[INFO] Ready! Running synthetic queries...")
    total_retrieval = 0
    total_gen = 0
    for i, question in enumerate(example_questions, 1):
        print(f"\n--- Example Query {i} ---")
        result = rag.answer_question(question, top_k)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:300]}...\n")
        print(f"Retrieval time: {result['retrieval_time']:.2f}s | Generation time: {result['generation_time']:.2f}s")
        total_retrieval += result['retrieval_time']
        total_gen += result['generation_time']
    print(f"\n[INFO] Average retrieval time: {total_retrieval/len(example_questions):.2f}s | Average generation time: {total_gen/len(example_questions):.2f}s")

if __name__ == "__main__":
    cli() 