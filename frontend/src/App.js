import React, { useRef, useState } from 'react';
import './App.css';
import { AnswerPanel } from './components/AnswerPanel';
import { QueryComposer } from './components/QueryComposer';
import { BrandMark, ChevronIcon } from './components/Icons';
import { PosterDecor } from './components/PosterDecor';

const API_BASE = process.env.REACT_APP_API_BASE || '/api';

const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिन्दी', englishName: 'Hindi' },
  { code: 'bn', name: 'বাংলা', englishName: 'Bengali' },
  { code: 'ta', name: 'தமிழ்', englishName: 'Tamil' },
  { code: 'te', name: 'తెలుగు', englishName: 'Telugu' },
  { code: 'mr', name: 'मराठी', englishName: 'Marathi' },
  { code: 'gu', name: 'ગુજરાતી', englishName: 'Gujarati' },
  { code: 'kn', name: 'ಕನ್ನಡ', englishName: 'Kannada' },
  { code: 'ml', name: 'മലയാളം', englishName: 'Malayalam' },
  { code: 'pa', name: 'ਪੰਜਾਬੀ', englishName: 'Punjabi' },
  { code: 'or', name: 'ଓଡ଼ିଆ', englishName: 'Odia' },
  { code: 'as', name: 'অসমীয়া', englishName: 'Assamese' },
  { code: 'ur', name: 'اردو', englishName: 'Urdu' },
  { code: 'ne', name: 'नेपाली', englishName: 'Nepali' },
  { code: 'sa', name: 'संस्कृतम्', englishName: 'Sanskrit' },
];

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState(null);
  const [requestState, setRequestState] = useState('idle');
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [language, setLanguage] = useState('en');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const isLoading = requestState !== 'idle';

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await transcribeAudio(audioBlob);
      };
      mediaRecorder.start(100);
      setIsRecording(true);
    } catch (err) {
      setError('Microphone access was denied or is not available on this device.');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
      setIsRecording(false);
    }
  };

  const transcribeAudio = async (audioBlob) => {
    try {
      setRequestState('transcribing');
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', language);
      const response = await fetch(`${API_BASE}/transcribe`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Transcription failed');
      const data = await response.json();
      setQuery(data.transcription);
    } catch (err) {
      setError('Transcription failed: ' + err.message);
    } finally {
      setRequestState('idle');
    }
  };

  const submitQuery = async () => {
    if (!query.trim()) return;
    try {
      setRequestState('querying');
      setError(null);
      setAnswer(null);
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, language, top_k: 5 }),
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Query failed');
      }
      setAnswer(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setRequestState('idle');
    }
  };

  return (
    <div className="app-shell">
      <PosterDecor />
      <header className="site-header">
        <a className="brand" href="#main" aria-label="Vaani home">
          <BrandMark /><span className="brand-name">Vaani</span><span className="brand-rule" aria-hidden="true" />
          <span className="brand-caption">Voice-powered knowledge from the coast</span>
        </a>
        <div className="header-meta" aria-label="Product capabilities"><span className="status-dot" aria-hidden="true" />15 languages · cited answers</div>
      </header>

      <main id="main" className="workspace">
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">Multilingual knowledge retrieval</p>
          <h1 id="page-title"><span>Ask out loud.</span><br />Know for sure.</h1>
          <p className="intro-copy">Speak or type a question. Vaani searches the knowledge index, reranks the strongest passages, and returns an answer you can trace back to its source.</p>
          <div className="flow-line" aria-label="How Vaani works"><span><b>01</b> Ask</span><ChevronIcon /><span><b>02</b> Retrieve</span><ChevronIcon /><span><b>03</b> Verify</span></div>
          <div className="hero-stamp" aria-hidden="true"><span>15</span><small>languages</small></div>
        </section>

        <section className="interaction-area" aria-label="Ask Vaani">
          <QueryComposer query={query} setQuery={setQuery} language={language} setLanguage={setLanguage} languages={languages} isLoading={isLoading} requestState={requestState} isRecording={isRecording} startRecording={startRecording} stopRecording={stopRecording} submitQuery={submitQuery} error={error} />
          {requestState === 'querying' && <div className="result-loading" role="status" aria-live="polite"><div className="loading-rule"><span /></div><div><strong>Searching the index</strong><p>Retrieving and verifying the most relevant passages…</p></div></div>}
          {answer ? <AnswerPanel answer={answer} /> : requestState !== 'querying' && <div className="empty-result" aria-label="Answer area"><span className="empty-index">A</span><div><h2>Your grounded answer will appear here</h2><p>Supporting passages and retrieval timing are included with every response.</p></div></div>}
        </section>
      </main>

      <footer className="site-footer"><span>Vaani · HackerHouse 2026</span><span>MSMARCO-XI / FAISS / FlashRank / Sarvam STT</span></footer>
    </div>
  );
}

export default App;
