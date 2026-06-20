import React, { useState, useEffect, useContext } from 'react';
import { SessionContext } from '../contexts/SessionContext';
import { getSessionStatus, refreshSession } from '../services/api';
import './SessionStatus.css';

const SessionStatus = () => {
  const { sessionId } = useContext(SessionContext);
  const [status, setStatus] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStatus = async () => {
    if (!sessionId) return;
    try {
      const statusData = await getSessionStatus(sessionId);
      setStatus(statusData);
    } catch (error) {
      console.error('Failed to fetch session status:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshSession(sessionId);
      await fetchStatus();
    } catch (error) {
      console.error('Failed to refresh session:', error);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 60000); // Check every minute
    return () => clearInterval(interval);
  }, [sessionId]);

  if (!status) return null;

  const isLowTime = status.remaining_minutes && status.remaining_minutes < 10;

  return (
    <div className={`session-status ${isLowTime ? 'warning' : ''}`}>
      <span className="session-info">
        ⏱️ {status.remaining_minutes ? `${Math.round(status.remaining_minutes)}m left` : 'Active'}
      </span>
      <button 
        className="refresh-btn" 
        onClick={handleRefresh}
        disabled={refreshing}
        title="Refresh session"
      >
        {refreshing ? '⏳' : '🔄'}
      </button>
    </div>
  );
};

export default SessionStatus;