import React from 'react';

const Icon = ({ children, size = 20 }) => <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">{children}</svg>;
export const BrandMark = () => <svg className="brand-mark" width="38" height="38" viewBox="0 0 38 38" fill="none" aria-hidden="true"><rect x="1" y="1" width="36" height="36" rx="10" stroke="currentColor" strokeWidth="2" /><path d="M10 19h3m3-6v12m4-16v20m4-15v10m4-7v4" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" /></svg>;
export const MicIcon = () => <Icon><rect x="9" y="3" width="6" height="11" rx="3" stroke="currentColor" strokeWidth="1.8" /><path d="M5.5 11.5a6.5 6.5 0 0013 0M12 18v3m-3 0h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></Icon>;
export const StopIcon = () => <Icon><rect x="6.5" y="6.5" width="11" height="11" rx="1.5" fill="currentColor" /></Icon>;
export const SearchIcon = () => <Icon><circle cx="10.8" cy="10.8" r="6.3" stroke="currentColor" strokeWidth="1.8" /><path d="M15.5 15.5L20 20" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></Icon>;
export const AlertIcon = () => <Icon><path d="M12 3L2.8 19h18.4L12 3z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" /><path d="M12 9v4m0 3v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></Icon>;
export const CheckIcon = () => <Icon size={16}><path d="M5 12.5l4 4L19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></Icon>;
export const FileIcon = () => <Icon size={18}><path d="M7 3h7l4 4v14H7V3z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" /><path d="M14 3v5h4M10 12h5m-5 4h5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></Icon>;
export const ClockIcon = () => <Icon size={18}><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.7" /><path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" /></Icon>;
export const ChevronIcon = () => <Icon size={14}><path d="M9 6l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></Icon>;
