import React from 'react';
import { AlertIcon, MicIcon, SearchIcon, StopIcon } from './Icons';

export function QueryComposer({ query, setQuery, language, setLanguage, languages, isLoading, requestState, isRecording, startRecording, stopRecording, submitQuery, error }) {
  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !isLoading && query.trim()) submitQuery();
  };
  return <div className={`composer ${isRecording ? 'is-recording' : ''}`}>
    <div className="composer-topline"><div><span className="section-kicker">Your question</span><h2>What would you like to know?</h2></div><label className="language-field" htmlFor="language"><span>Answer language</span><select id="language" value={language} onChange={(event) => setLanguage(event.target.value)} disabled={isLoading || isRecording}>{languages.map((item) => <option key={item.code} value={item.code}>{item.name}{item.englishName ? ` · ${item.englishName}` : ''}</option>)}</select></label></div>
    <div className="query-field"><textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask a question by typing here…" rows={4} maxLength={2000} disabled={isLoading || isRecording} aria-describedby="query-hint" /><div className="field-meta" id="query-hint"><span>{isRecording ? 'Listening… speak clearly in your selected language' : 'Speak naturally or type your question'}</span><span>{query.length} / 2000</span></div></div>
    {isRecording && <div className="recording-strip" role="status" aria-live="polite"><span className="recording-pulse" /><strong>Recording</strong><span className="waveform" aria-hidden="true">{[1,2,3,4,5,6,7,8,9,10,11,12].map((bar) => <i key={bar} />)}</span></div>}
    <div className="composer-actions"><button className={`button button-voice ${isRecording ? 'active' : ''}`} type="button" onClick={isRecording ? stopRecording : startRecording} disabled={isLoading}>{isRecording ? <StopIcon /> : <MicIcon />}{isRecording ? 'Stop & transcribe' : 'Ask with voice'}</button><button className="button button-submit" type="button" onClick={submitQuery} disabled={isLoading || !query.trim()}>{requestState === 'querying' ? <span className="spinner" /> : <SearchIcon />}{requestState === 'querying' ? 'Finding an answer' : requestState === 'transcribing' ? 'Transcribing audio' : 'Find grounded answer'}</button></div>
    {error && <div className="error-message" role="alert"><AlertIcon /><div><strong>We couldn’t complete that request</strong><span>{error}</span></div></div>}
  </div>;
}
