import React from 'react';
import './PipelineStages.css';

const STAGES = [
  {
    id: 'pdf',
    icon: '📄',
    label: 'PDF',
    description: 'Source documents',
  },
  {
    id: 'chunking',
    icon: '✂️',
    label: 'Chunking',
    description: 'Split into passages',
  },
  {
    id: 'embedding',
    icon: '🔢',
    label: 'Embedding',
    description: 'Dense + sparse vectors',
  },
  {
    id: 'vector-store',
    icon: '🗄️',
    label: 'Vector Store',
    description: 'Persistent Qdrant index',
  },
];

function PipelineStages() {
  return (
    <section
      className="pipeline-stages"
      aria-label="Ingestion pipeline stages"
    >
      <h3 className="pipeline-title">Ingestion Pipeline</h3>
      <ol className="pipeline-flow" role="list">
        {STAGES.map((stage, index) => (
          <React.Fragment key={stage.id}>
            <li className="pipeline-stage" aria-label={`Stage ${index + 1}: ${stage.label}`}>
              <span className="stage-icon" aria-hidden="true">{stage.icon}</span>
              <span className="stage-label">{stage.label}</span>
              <span className="stage-description">{stage.description}</span>
            </li>
            {index < STAGES.length - 1 && (
              <li className="pipeline-arrow" aria-hidden="true" role="presentation">
                <span className="arrow-line" />
                <span className="arrow-head">›</span>
              </li>
            )}
          </React.Fragment>
        ))}
      </ol>
    </section>
  );
}

export default PipelineStages;
