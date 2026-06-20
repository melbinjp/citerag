import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import './LanguageSwitcher.css';

const languageMap = {
  en: {
    short: 'EN',
    full: 'English',
  },
  es: {
    short: 'ES',
    full: 'Español',
  },
};

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    setIsOpen(false);
  };

  return (
    <div className="language-switcher">
      <button className="language-switcher-button" onClick={() => setIsOpen(!isOpen)}>
        <span className="globe-icon">🌐</span>
        {!isOpen && <span className="language-short">{languageMap[i18n.language]?.short}</span>}
        {isOpen && <span className="language-full">{languageMap[i18n.language]?.full}</span>}
      </button>
      {isOpen && (
        <div className="language-options">
          <button onClick={() => changeLanguage('en')}>English</button>
          <button onClick={() => changeLanguage('es')}>Español</button>
        </div>
      )}
    </div>
  );
};

export default LanguageSwitcher;
