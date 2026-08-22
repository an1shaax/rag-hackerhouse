import React from 'react';
import { CheckIcon, ClockIcon, FileIcon } from './Icons';

const latencyRows = [['Embedding', 'query_embedding_ms'], ['Retrieval', 'retrieval_ms'], ['Reranking', 'reranking_ms'], ['Generation', 'generation_ms'], ['Grounding', 'grounding_ms'], ['Guardrails', 'guardrails_ms']];
const formatMs = (value) => typeof value === 'number' ? `${value.toFixed(1)} ms` : '—';

export function AnswerPanel({ answer }) {
  const citations = answer.citations || [];
  return <article className="answer-panel" aria-labelledby="answer-title">
    <header className="answer-heading"><div><span className="section-kicker">Response</span><h2 id="answer-title">Grounded answer</h2></div><div className="answer-badges">{answer.refused && <span className="badge badge-refused">Refused</span>}{answer.grounded && <span className="badge badge-grounded"><CheckIcon /> Grounded</span>}</div></header>
    <div className="answer-body"><p>{answer.answer}</p></div>
    <div className="answer-facts"><div><span>Confidence</span><strong>{(answer.confidence * 100).toFixed(0)}%</strong></div><div><span>Language</span><strong>{answer.language?.toUpperCase()}</strong></div><div><span>Evidence</span><strong>{citations.length} {citations.length === 1 ? 'source' : 'sources'}</strong></div><div><span>Total time</span><strong>{formatMs(answer.latency?.total_rag_ms)}</strong></div></div>
    {citations.length > 0 && <section className="evidence-section" aria-labelledby="evidence-title"><div className="subsection-title"><FileIcon /><h3 id="evidence-title">Supporting evidence</h3><span>{citations.length}</span></div><div className="citation-list">{citations.map((citation, index) => <article className="citation" key={citation.chunk_id || `${citation.source}-${index}`}><span className="citation-number">{String(index + 1).padStart(2, '0')}</span><div><div className="citation-meta"><strong>{citation.source}</strong><span>Relevance {citation.score.toFixed(3)}</span></div><p>{citation.text_preview}</p></div></article>)}</div></section>}
    {answer.latency && <details className="latency-details"><summary><span><ClockIcon /> Retrieval timing</span><strong>{formatMs(answer.latency.total_rag_ms)}</strong></summary><div className="latency-grid">{latencyRows.map(([label, key]) => <div key={key}><span>{label}</span><strong>{formatMs(answer.latency[key])}</strong></div>)}</div></details>}
  </article>;
}
