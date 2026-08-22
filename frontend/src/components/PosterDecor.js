import React from 'react';

export function PosterDecor() {
  return <div className="poster-decor" aria-hidden="true">
    <svg className="decor decor-palm" viewBox="0 0 260 310" fill="none">
      <path d="M53 303c43-73 56-148 54-231" stroke="currentColor" strokeWidth="6" strokeLinecap="round" />
      <path d="M108 77C73 70 35 84 9 111c42 3 72-8 99-34Zm2-3C75 49 57 24 57 6c35 13 52 34 53 68Zm3 2c8-39 30-61 60-72-4 31-25 58-60 72Zm2 3c38-16 75-11 105 10-36 16-72 13-105-10Zm-10 5c-35 12-59 34-72 67 37-8 60-30 72-67Z" fill="currentColor" />
      <path d="M62 279c31-24 65-32 102-22-24 27-58 34-102 22Zm8-38c-5-34 4-62 28-84 13 33 3 61-28 84Z" fill="currentColor" opacity=".72" />
    </svg>
    <svg className="decor decor-flower" viewBox="0 0 180 180" fill="none">
      <path d="M90 79C45 31 59 5 85 44 85 8 112 3 101 48c34-31 54-8 15 27 47-5 48 23 3 20 39 28 19 51-16 18 10 46-18 49-18 5-27 39-53 19-17-18-43 5-52-21-11-2-1-47-29-18-45-4-25-38 22-23-5-49 23-10 24-46 39-12 5-48 32-40 26 7Z" fill="currentColor" />
      <circle cx="90" cy="82" r="19" fill="var(--forest)" stroke="var(--sun)" strokeWidth="5" />
    </svg>
    <svg className="decor decor-wave" viewBox="0 0 760 170" preserveAspectRatio="none">
      <path d="M0 102c95-98 161 77 256-14s155 76 258-8 152 52 246-22v112H0V102Z" fill="currentColor" />
      <path d="M0 125c97-69 171 58 267-10s160 43 254-9 151 27 239-20" stroke="var(--pink)" strokeWidth="7" fill="none" />
    </svg>
    <div className="decor-dots">{Array.from({ length: 9 }, (_, index) => <i key={index} />)}</div>
  </div>;
}
