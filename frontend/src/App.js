import React, { useState, useRef } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || '/api';

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [language, setLanguage] = useState('en');
  const [latency, setLatency] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिन्दी (Hindi)' },
    { code: 'bn', name: 'বাংলা (Bengali)' },
    { code: 'ta', name: 'தமிழ் (Tamil)' },
    { code: 'te', name: 'తెలుగు (Telugu)' },
    { code: 'mr', name: 'मराठी (Marathi)' },
    { code: 'gu', name: 'ગુજરાતી (Gujarati)' },
    { code: 'kn', name: 'ಕನ್ನಡ (Kannada)' },
    { code: 'ml', name: 'മലയാളം (Malayalam)' },
    { code: 'pa', name: 'ਪੰਜਾਬੀ (Punjabi)' },
    { code: 'or', name: 'ଓଡ଼ିଆ (Odia)' },
    { code: 'as', name: 'অসমীয়া (Assamese)' },
    { code: 'ur', name: 'اردو (Urdu)' },
    { code: 'ne', name: 'नेपाली (Nepali)' },
    { code: 'sa', name: 'संस्कृतम् (Sanskrit)' },
  ];

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);
      };

      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access denied or not available');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      setIsRecording(false);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', language);

      const response = await fetch(`${API_BASE}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Transcription failed');
      }

      const data = await response.json();
      setQuery(data.transcription);
    } catch (err) {
      setError('Transcription failed: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const submitQuery = async () => {
    if (!query.trim()) return;

    try {
      setIsLoading(true);
      setError(null);
      setAnswer(null);
      setLatency(null);

      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, language, top_k: 5 }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Query failed');
      }

      const data = await response.json();
      setAnswer(data);
      setLatency(data.latency);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const formatLatency = (latency) => {
    if (!latency) return null;
    return (
      <div className="latency-breakdown">
        <h4>Latency Breakdown</h4>
        <table>
          <tbody>
            <tr><td>Query Embedding</td><td>{latency.query_embedding_ms?.toFixed(1)} ms</td></tr>
            <tr><td>Retrieval</td><td>{latency.retrieval_ms?.toFixed(1)} ms</td></tr>
            <tr><td>Reranking</td><td>{latency.reranking_ms?.toFixed(1)} ms</td></tr>
            <tr><td>Generation</td><td>{latency.generation_ms?.toFixed(1)} ms</td></tr>
            <tr><td>Grounding</td><td>{latency.grounding_ms?.toFixed(1)} ms</td></tr>
            <tr><td>Guardrails</td><td>{latency.guardrails_ms?.toFixed(1)} ms</td></tr>
            <tr className="total"><td><strong>Total RAG</strong></td><td><strong>{latency.total_rag_ms?.toFixed(1)} ms</strong></td></tr>
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🎤 Voice-Enabled RAG System</h1>
        <p className="subtitle">HackerHouse 2026 - MSMARCO-XI Multilingual QA</p>
      </header>

      <main className="app-main">
        <div className="card query-card">
          <div className="language-selector">
            <label htmlFor="language">Language: </label>
            <select id="language" value={language} onChange={(e) => setLanguage(e.target.value)}>
              {languages.map((lang) => (
                <option key={lang.code} value={lang.code}>{lang.name}</option>
              ))}
            </select>
          </div>

          <div className="input-group">
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type your question or use voice input..."
              rows={3}
              disabled={isLoading || isRecording}
            />
            <div className="input-actions">
              <button
                className={isRecording ? 'btn btn-primary recording' : 'btn btn-primary'}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
              >
                {isRecording ? '🔴 Stop Recording' : '🎤 Start Recording'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={submitQuery}
                disabled={isLoading || !query.trim()}
              >
                {isLoading ? '⏳ Processing...' : '🔍 Search & Answer'}
              </button>
            </div>
          </div>

          {error && <div className="error">{error}</div>}
        </div>

        {answer && (
          <div className="card answer-card">
            <div className="answer-header">
              <h3>Answer</h3>
              {answer.refused && <span className="badge refused">Refused</span>}
              {answer.grounded && <span className="badge grounded">Grounded</span>}
            </div>
            <div className="answer-content">
              <p>{answer.answer}</p>
              <div className="answer-meta">
                <span>Confidence: {(answer.confidence * 100).toFixed(0)}%</span>
                <span>Language: {answer.language}</span>
              </div>
            </div>

            {answer.citations && answer.citations.length > 0 && (
              <div className="citations">
                <h4>Citations</h4>
                <ul>
                  {answer.citations.map((citation, idx) => (
                    <li key={idx}>
                      <strong>{citation.source}</strong> (score: {citation.score.toFixed(3)})
                      <br />
                      <small>{citation.text_preview}</small>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {formatLatency(latency)}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Built for HackerHouse 2026 | Powered by MSMARCO-XI, FAISS, FlashRank, Sarvam STT</p>
      </footer>
    </div>
  );
}

export default App;