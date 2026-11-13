"""Lab: Re-ranking a RAG pipeline.
 
Two-stage retrieval over SciFact, a set of 5,183 biomedical abstracts and 300
test claims with known relevant abstracts. Stage 1 (given) is a fast bi-encoder
search. Stage 2 is a cross-encoder that re-scores the stage 1 candidates.
"""
 
# ======================= WRITTEN ANSWERS =======================
#
# STEP 5
# Why does build_index() put the title in front of every chunk?
#   An abstract is split into chunks of about 120 words. A chunk from the
#   middle or end of an abstract often does not name what the paper is about
#   (it says "the treatment" or "these patients"). Adding the title gives every
#   chunk its topic, so its embedding lands close to claims about that topic.
#
# Why is to_doc_ids() needed?
#   Stage 1 returns chunks, but the answer key and the LLM prompt work with
#   whole abstracts. Several chunks in the results can come from the same
#   abstract. to_doc_ids() maps each chunk back to its abstract and keeps each
#   abstract only once, at the position of its best chunk. Without it the same
#   abstract could fill several of the top k places and the metrics would be
#   scored on chunks instead of abstracts.
#
# STEP 10  (FILL IN the numbers from your own runs, k = 3, 100 claims)
#   Candidates | Recall@3 before -> after | MRR before -> after | re-rank ms
#   5          | ____ -> ____             | ____ -> ____        | ____
#   10         | ____ -> ____             | ____ -> ____        | ____
#   30         | ____ -> ____             | ____ -> ____        | ____
#   100        | ____ -> ____             | ____ -> ____        | ____
#
#   Why re-ranking helps so little at 5 candidates:
#   The re-ranker never searches. It can only re-order the chunks stage 1
#   gave it. With 5 chunks (often only 3 or 4 different abstracts) there is
#   almost nothing to re-order: if the right abstract is not among those 5,
#   the re-ranker cannot bring it back, and if it is, it is usually already
#   in or near the top 3.
#
#   What it costs at 100 candidates:
#   The cross-encoder reads every (claim, chunk) pair, so 100 candidates is
#   about 20 times the work of 5. Re-rank time grew from ____ ms to ____ ms
#   per claim, while Recall@3 and MRR only improved by ____ and ____ compared
#   with 30 candidates. Most of the benefit arrives by ____ candidates; after
#   that I am paying latency for very little extra quality.
#
# STEP 13  (FILL IN from the claim you find)
#   Claim: "____"
#   Without re-ranking the model answered ____ and cited [____].
#   With re-ranking the model answered ____ and cited [____].
#   The answer key lists abstract [____], which was / was not in each top 3.
#   The ____ answer is right because ____.
#   What made the difference: re-ranking moved abstract [____] into the top 3
#   (or pushed it out), so the model saw different evidence. The re-ranker
#   only scores how relevant an abstract is to the claim, not whether it
#   agrees with it, so the abstract it promoted can be one that refutes the
#   claim. The verdict changed because the evidence changed, not because the
#   model changed.
# ===============================================================
 
import json
import time
 
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer
 
DATA_DIR = "data/scifact"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHAT_MODEL = "gpt-4.1-mini"
 
CHUNK_WORDS = 120    # words per chunk
CHUNK_OVERLAP = 30   # words shared by neighbouring chunks
CANDIDATES = 30      # chunks stage 1 hands to the reranker
CONTEXT_K = 3        # abstracts that go into the prompt
EVAL_QUERIES = 100   # test claims used in Part C
 
# Part D: the system prompt
SYSTEM_TEMPLATE = """You are a careful scientific fact-checker.
You will be given a claim. Decide whether the research abstracts below support it.
 
Rules:
1. Use ONLY the abstracts below. Do not use any outside knowledge, even if you
  believe you know the answer.
2. Begin your reply with exactly one verdict, on its own line:
  SUPPORTED, REFUTED, or NOT ENOUGH EVIDENCE.
  - SUPPORTED: the abstracts show the claim is true.
  - REFUTED: the abstracts show the claim is false.
  - NOT ENOUGH EVIDENCE: the abstracts do not address the claim, or do not
    settle it.
3. Then explain your verdict in two or three sentences. Cite every abstract
  you relied on by its id in square brackets, for example [12345]. Cite only
  ids that appear below.
 
Abstracts:
{context}"""
 
 
# ---------- Given: loading ----------
@st.cache_data
def load_scifact(data_dir=DATA_DIR):
   """Return (corpus, queries, qrels).
 
   corpus:  doc_id -> {"title", "text"}
   queries: query_id -> claim text (the 300 test claims)
   qrels:   query_id -> set of relevant doc_ids (the answer key)
   """
   corpus = {}
   with open(f"{data_dir}/corpus.jsonl", encoding="utf-8") as f:
       for line in f:
           row = json.loads(line)
           corpus[row["_id"]] = {"title": row["title"], "text": row["text"]}
 
   queries = {}
   with open(f"{data_dir}/queries.jsonl", encoding="utf-8") as f:
       for line in f:
           row = json.loads(line)
           queries[row["_id"]] = row["text"]
 
   qrels = {}
   with open(f"{data_dir}/qrels/test.tsv", encoding="utf-8") as f:
       next(f)  # header
       for line in f:
           query_id, doc_id, _ = line.strip().split("\t")
           qrels.setdefault(query_id, set()).add(doc_id)
 
   return corpus, queries, qrels
 
 

def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
   """Split text into windows of `size` words that overlap by `overlap` words."""
   words = text.split()
   step = size - overlap
   starts = range(0, max(len(words) - overlap, 1), step)
   return [" ".join(words[start:start + size]) for start in starts]
 
 
# ---------- Given: embedding ----------
@st.cache_resource
def load_embedder():
   return SentenceTransformer(EMBED_MODEL)
 
 
@st.cache_resource
def build_index(size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
   """Chunk every abstract, embed every chunk, keep a chunk -> doc_id map."""
   corpus, _, _ = load_scifact()
   doc_ids, texts = [], []
   for doc_id, doc in corpus.items():
       for piece in chunk_text(doc["text"], size, overlap):
           doc_ids.append(doc_id)
           texts.append(f"{doc['title']}. {piece}")
 
   vectors = load_embedder().encode(
       texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
   )
   return {"doc_ids": doc_ids, "texts": texts, "vectors": vectors}
 
 
# ---------- Given: stage 1 retrieval ----------
def retrieve(index, query, k=CANDIDATES):
   """Return the top k chunks as [(chunk_index, cosine_score), ...]."""
   query_vector = load_embedder().encode(query, normalize_embeddings=True)
   scores = index["vectors"] @ query_vector
   top = np.argsort(-scores)[:k]
   return [(int(i), float(scores[i])) for i in top]
 
 
def to_doc_ids(index, hits):
   """Collapse ranked chunks into ranked abstracts (first chunk wins)."""
   ranked = []
   for chunk_index, _ in hits:
       doc_id = index["doc_ids"][chunk_index]
       if doc_id not in ranked:
           ranked.append(doc_id)
   return ranked
 
 
# ---------- Part B: stage 2 re-ranking ----------
@st.cache_resource
def load_reranker():
   return CrossEncoder(RERANK_MODEL)
 
 
def rerank(index, query, hits):
   """Re-score stage 1 hits with the cross-encoder, best first."""
   if not hits:
       return []
 
   # One (claim, chunk text) pair for every hit
   pairs = [(query, index["texts"][chunk_index]) for chunk_index, _ in hits]
 
   # Score every pair in a single predict() call (no loop)
   scores = load_reranker().predict(pairs)
 
   # Same (chunk_index, score) format as stage 1, now with the reranker score
   reranked = [
       (chunk_index, float(score))
       for (chunk_index, _), score in zip(hits, scores)
   ]
   reranked.sort(key=lambda hit: hit[1], reverse=True)  # highest first
   return reranked
 
 
# ---------- Given: metrics ----------
def recall_at_k(ranked, relevant, k):
   """Share of the relevant abstracts that appear in the top k."""
   return len(set(ranked[:k]) & relevant) / len(relevant)
 
 
def reciprocal_rank(ranked, relevant):
   """1 / rank of the first relevant abstract (0 if none was retrieved)."""
   for rank, doc_id in enumerate(ranked, start=1):
       if doc_id in relevant:
           return 1 / rank
   return 0.0
 
 
# ---------- Part C: measure before and after ----------
def evaluate(index, n_queries=EVAL_QUERIES, candidates=CANDIDATES, k=CONTEXT_K):
   """Average recall@k, MRR and latency, with and without re-ranking."""
   _, queries, qrels = load_scifact()
   query_ids = list(qrels)[:n_queries]
 
   recall_before = recall_after = 0.0
   rr_before = rr_after = 0.0
   retrieve_seconds = rerank_seconds = 0.0
 
   for query_id in query_ids:
       claim = queries[query_id]
       relevant = qrels[query_id]
 
       # Stage 1, timed by itself
       start = time.perf_counter()
       hits = retrieve(index, claim, candidates)
       retrieve_seconds += time.perf_counter() - start
 
       # Stage 2, timed by itself
       start = time.perf_counter()
       reranked = rerank(index, claim, hits)
       rerank_seconds += time.perf_counter() - start
 
       # Chunks -> abstracts, then score both rankings against the answer key
       before = to_doc_ids(index, hits)
       after = to_doc_ids(index, reranked)
 
       recall_before += recall_at_k(before, relevant, k)
       recall_after += recall_at_k(after, relevant, k)
       rr_before += reciprocal_rank(before, relevant)
       rr_after += reciprocal_rank(after, relevant)
 
   n = len(query_ids)
   return {
       "recall_before": recall_before / n,
       "recall_after": recall_after / n,
       "rr_before": rr_before / n,
       "rr_after": rr_after / n,
       "retrieve_ms": retrieve_seconds / n * 1000,
       "rerank_ms": rerank_seconds / n * 1000,
   }
 
 
# ---------- Part D: generation ----------
def build_context(doc_ids):
   corpus, _, _ = load_scifact()
   return "\n\n".join(
       f"[{doc_id}] {corpus[doc_id]['title']}\n{corpus[doc_id]['text']}"
       for doc_id in doc_ids
   )
 
 
def verify_claim(claim, doc_ids):
   """Ask CHAT_MODEL for a verdict on the claim, using only the given abstracts."""
   # Same client setup as Lab 4
   if "openai_client" not in st.session_state:
       st.session_state.openai_client = OpenAI(
           api_key=st.secrets["OPENAI_KEY"]
       )
   client = st.session_state.openai_client
 
   system_prompt = SYSTEM_TEMPLATE.format(context=build_context(doc_ids))
 
   response = client.chat.completions.create(
       model=CHAT_MODEL,
       temperature=0,  # so the two answers differ only because the evidence differs
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": f"Claim: {claim}"},
       ],
   )
   return response.choices[0].message.content
 
 
# ---------- UI ----------
def show_docs(doc_ids, relevant):
   corpus, _, _ = load_scifact()
   for rank, doc_id in enumerate(doc_ids, start=1):
       mark = " ✅" if doc_id in relevant else ""
       st.markdown(f"**{rank}. [{doc_id}]**{mark} {corpus[doc_id]['title']}")
 
 
def main():
   st.title("Lab: Re-ranking a RAG Pipeline")
   corpus, queries, qrels = load_scifact()
 
   candidates = st.sidebar.slider("Stage 1 candidates", 5, 100, CANDIDATES, step=5)
   k = st.sidebar.slider("Abstracts sent to the LLM", 1, 5, CONTEXT_K)
 
   with st.spinner("Chunking and embedding 5,183 abstracts (first run only)..."):
       index = build_index()
   st.caption(f"{len(corpus):,} abstracts, {len(index['texts']):,} chunks indexed")
 
   # ----- Part A: pick a claim, look at stage 1 -----
   test_ids = list(qrels)
   query_id = st.selectbox(
       "Pick a test claim", test_ids, format_func=lambda qid: queries[qid]
   )
   custom = st.text_input("...or type your own claim (no answer key)")
   claim = custom or queries[query_id]
   relevant = set() if custom else qrels[query_id]
 
   hits = retrieve(index, claim, candidates)
   before = to_doc_ids(index, hits)
 
   # ----- Part B: side by side -----
   start = time.perf_counter()
   after = to_doc_ids(index, rerank(index, claim, hits))
   rerank_ms = (time.perf_counter() - start) * 1000
 
   left, right = st.columns(2)
   with left:
       st.subheader("Without re-ranking")
       show_docs(before[:k], relevant)
   with right:
       st.subheader("With re-ranking")
       show_docs(after[:k], relevant)
   st.caption(f"Re-ranking {candidates} candidates took {rerank_ms:.0f} ms. ✅ = in the answer key.")
 
   # ----- Part C: evaluation button -----
   st.divider()
   st.subheader("Part C: Measure before and after")
   if st.button("Run evaluation"):
       with st.spinner(f"Scoring {EVAL_QUERIES} claims with {candidates} candidates..."):
           results = evaluate(index, EVAL_QUERIES, candidates, k)
 
       recall_change = results["recall_after"] - results["recall_before"]
       mrr_change = results["rr_after"] - results["rr_before"]
       total_ms = results["retrieve_ms"] + results["rerank_ms"]
 
       col1, col2, col3 = st.columns(3)
       col1.metric(
           f"Recall@{k}",
           f"{results['recall_after']:.3f}",
           delta=f"{recall_change:+.3f}",
       )
       col2.metric(
           "MRR",
           f"{results['rr_after']:.3f}",
           delta=f"{mrr_change:+.3f}",
       )
       col3.metric(
           "Latency per claim",
           f"{total_ms:.0f} ms",
           delta=f"+{results['rerank_ms']:.0f} ms from re-ranking",
           delta_color="inverse",  # extra time is a cost, so show it in red
       )
       st.caption(
           f"{EVAL_QUERIES} claims, {candidates} stage 1 candidates. "
           f"Before re-ranking: Recall@{k} {results['recall_before']:.3f}, "
           f"MRR {results['rr_before']:.3f}, "
           f"stage 1 alone {results['retrieve_ms']:.0f} ms per claim."
       )
 
   # ----- Part D: two answers side by side -----
   st.divider()
   st.subheader("Part D: Does re-ranking change the answer?")
   if st.button("Verify claim"):
       with st.spinner("Asking the model twice..."):
           answer_before = verify_claim(claim, before[:k])
           answer_after = verify_claim(claim, after[:k])
 
       left, right = st.columns(2)
       with left:
           st.markdown("**Answer without re-ranking**")
           st.write(answer_before)
           st.caption("Abstracts used: " + ", ".join(f"[{d}]" for d in before[:k]))
       with right:
           st.markdown("**Answer with re-ranking**")
           st.write(answer_after)
           st.caption("Abstracts used: " + ", ".join(f"[{d}]" for d in after[:k]))
 
 
if __name__ == "__main__":
   main()
