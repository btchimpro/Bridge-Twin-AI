import { useState, useEffect, Suspense } from 'react';
import { AlertTriangle, Activity, Wind, Car, ShieldAlert, Cpu, Wrench, ChevronRight } from 'lucide-react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, Grid, ContactShadows } from '@react-three/drei';
import { useBridgeContext } from '../context/BridgeContext';
import { apiClient } from '../api/client';

// ---------------------------------------------------------
// 3D Model Placeholder
// ---------------------------------------------------------
const BridgeModelPlaceholder = ({ type }: { type: string }) => {
  return (
    <mesh position={[0, 1, 0]}>
      <boxGeometry args={[10, 0.5, 2]} />
      <meshStandardMaterial color="#475569" metalness={0.5} roughness={0.2} />
      {/* Pillars */}
      <mesh position={[-3, -2, 0]}>
        <cylinderGeometry args={[0.3, 0.3, 4]} />
        <meshStandardMaterial color="#64748b" />
      </mesh>
      <mesh position={[3, -2, 0]}>
        <cylinderGeometry args={[0.3, 0.3, 4]} />
        <meshStandardMaterial color="#64748b" />
      </mesh>
      {type === 'cable-stayed' && (
        <mesh position={[0, 4, 0]}>
          <cylinderGeometry args={[0.2, 0.2, 6]} />
          <meshStandardMaterial color="#94a3b8" />
        </mesh>
      )}
    </mesh>
  );
};

// ---------------------------------------------------------
// Dashboard Page Component
// ---------------------------------------------------------
const DashboardPage = () => {
  const { selectedBridgeId } = useBridgeContext();
  const [loading, setLoading] = useState(false);
  const [assessment, setAssessment] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeLayer, setActiveLayer] = useState<'stress' | 'traffic' | 'weather'>('stress');

  useEffect(() => {
    if (!selectedBridgeId) return;

    const fetchAssessment = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data } = await apiClient.get(`/api/bridges/${selectedBridgeId}/assessment`);
        setAssessment(data);
      } catch (err: any) {
        setError(err.message || "Failed to load bridge assessment.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchAssessment();
  }, [selectedBridgeId]);

  if (!selectedBridgeId) {
    return <div className="p-gutter text-dark-text-muted">Please select a bridge from the header to view its dashboard.</div>;
  }

  if (error) {
    return (
      <div className="p-gutter h-full flex items-center justify-center">
        <div className="bg-error-container/20 border border-error-container/40 text-error p-4 rounded flex items-center gap-3">
          <AlertTriangle size={20} />
          {error}
        </div>
      </div>
    );
  }

  const risk = assessment?.risk_assessment;
  const shi = assessment?.structural_health;
  const telemetry = assessment?.bridge?.telemetry?.latest;
  const llm = assessment?.llm_analysis;
  const xai = assessment?.explainability;
  const bridgeName = assessment?.bridge?.name || 'Loading...';

  return (
    <div className="flex flex-col w-full p-gutter lg:p-gutter-lg gap-space-lg select-none text-on-surface pb-12">
      
      {/* Sub-Header / Page Level Engineering Status Bar */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-space-md pb-space-xs">
        <div className="flex flex-col gap-1">
          <div className="flex flex-wrap items-center gap-space-md">
            <h1 className="font-headline-lg text-headline-lg font-semibold tracking-tight text-dark-text-primary">
              Infrastructure Twin: {bridgeName}
            </h1>
            <span className="inline-flex items-center gap-1.5 px-space-sm py-0.5 rounded bg-status-optimal/10 text-status-optimal font-label-micro text-label-micro uppercase tracking-widest font-semibold">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-optimal opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-status-optimal"></span>
              </span>
              System Health: Optimal
            </span>
          </div>
          <div className="flex items-center gap-space-md text-dark-text-muted font-label-micro text-label-micro">
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-status-optimal"></span>
              Telemetry Online • Green Pulse Active
            </span>
            <span>•</span>
            <span className="font-mono">Live Sync</span>
            <span>•</span>
            <span className="text-on-surface-variant font-mono">ASSET ID: #{assessment?.bridge?.id}</span>
          </div>
        </div>

        {/* Quick Analytics Indicators & Global Actions */}
        <div className="flex flex-wrap items-center gap-space-sm">
          {/* LSTM SHI Metric Tile */}
          <div className="bg-surface-container-low px-space-md py-space-xs rounded flex items-center gap-space-md border border-dark-border-subtle">
            <div className="flex flex-col">
              <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">LSTM SHI</span>
              <span className="font-label-telemetry text-label-telemetry font-mono font-semibold text-primary">
                {loading ? '--' : (shi?.predicted_shi * 100).toFixed(1)}% 
                <span className="text-status-optimal font-body-sm text-body-sm font-normal ml-1">Stable</span>
              </span>
            </div>
            <svg className="w-12 h-6 text-primary" fill="none" viewBox="0 0 48 24">
              <path d="M1 20L10 16L18 18L26 9L34 11L42 3L47 5" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"></path>
            </svg>
          </div>
          {/* XGBoost Risk Tile */}
          <div className="bg-surface-container-low px-space-md py-space-xs rounded flex items-center gap-space-sm border border-dark-border-subtle">
            <div className={`w-1 h-8 rounded ${risk?.is_at_risk ? 'bg-status-critical' : 'bg-status-optimal'}`}></div>
            <div className="flex flex-col">
              <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">XGBoost Risk</span>
              <span className={`font-label-telemetry text-label-telemetry font-mono font-semibold ${risk?.is_at_risk ? 'text-status-critical' : 'text-status-optimal'}`}>
                {loading ? 'Analyzing' : risk?.is_at_risk ? 'High Risk' : 'Low Risk'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main 12-Column Responsive Dashboard Chassis */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-lg items-start">
        
        {/* Left Master Column: 3D Twin & Analytical Sub-Panels (8 Cols on Desktop) */}
        <div className="xl:col-span-8 flex flex-col gap-space-lg">
          
          {/* Digital Twin 3D Viewport Chassis */}
          <div className="relative w-full h-[470px] lg:h-[520px] rounded-lg bg-surface-container-lowest border border-dark-border-subtle overflow-hidden flex flex-col justify-between">
            <div className="absolute inset-0 z-0">
               <Suspense fallback={<div className="w-full h-full flex items-center justify-center text-dark-text-muted text-sm">Loading Model...</div>}>
                  <Canvas camera={{ position: [5, 5, 10], fov: 45 }}>
                    <color attach="background" args={['#0e0e13']} />
                    <ambientLight intensity={0.5} />
                    <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffffff" castShadow />
                    <Environment preset="city" />
                    <BridgeModelPlaceholder type={assessment?.bridge?.structure_type || 'girder'} />
                    <Grid infiniteGrid fadeDistance={40} fadeStrength={5} cellColor="#333333" sectionColor="#1a1a1a" />
                    <ContactShadows position={[0, -2.5, 0]} opacity={0.6} scale={20} blur={2} far={4} color="#000000" />
                    <OrbitControls makeDefault minPolarAngle={0} maxPolarAngle={Math.PI / 2 - 0.1} />
                  </Canvas>
                </Suspense>
            </div>
            
            {/* Viewport Top Overlay: Layer Selector & Spatial Modes */}
            <div className="relative z-10 p-space-md flex flex-wrap items-center justify-between gap-space-sm pointer-events-none">
              <div className="flex items-center gap-space-xs pointer-events-auto">
                <button className="px-space-md py-1.5 rounded bg-surface-container-highest/90 backdrop-blur-md text-dark-text-primary font-body-sm text-body-sm font-semibold flex items-center gap-1.5 shadow-sm border border-dark-border-subtle hover:bg-surface-bright transition-all">
                  <span className="material-symbols-outlined text-primary text-[18px]">layers</span>
                  <span>Layer: Structural BIM</span>
                </button>
              </div>
              
              {/* Multimodal Stress/Traffic/Weather Pill Controls */}
              <div className="flex items-center bg-surface-container-lowest/85 backdrop-blur-md p-1 rounded gap-1 shadow-md border border-dark-border-subtle pointer-events-auto">
                <button 
                  onClick={() => setActiveLayer('stress')}
                  className={`px-space-md py-1 rounded font-label-micro text-label-micro uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'stress' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full bg-accent-traffic ${activeLayer === 'stress' ? 'shadow-[0_0_8px_rgba(249,115,22,0.8)]' : ''}`}></span>
                  Stress
                </button>
                <button 
                  onClick={() => setActiveLayer('traffic')}
                  className={`px-space-md py-1 rounded font-label-micro text-label-micro uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'traffic' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full bg-telemetry-cyan ${activeLayer === 'traffic' ? 'shadow-[0_0_8px_rgba(6,182,212,0.8)]' : ''}`}></span>
                  Traffic
                </button>
                <button 
                  onClick={() => setActiveLayer('weather')}
                  className={`px-space-md py-1 rounded font-label-micro text-label-micro uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'weather' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-outline"></span>
                  Weather
                </button>
              </div>
            </div>

            {/* Viewport Bottom HUD Control Bar */}
            <div className="relative z-10 p-space-sm mx-space-md mb-space-md rounded bg-surface-container-lowest/90 backdrop-blur-md flex flex-wrap items-center justify-between gap-space-md shadow-xl border border-dark-border-subtle pointer-events-auto">
              <div className="flex items-center gap-1">
                <button className="px-space-md py-1 rounded hover:bg-surface-container-high text-on-surface-variant hover:text-dark-text-primary text-body-sm font-body-sm flex items-center gap-1 transition-colors">
                  <span className="material-symbols-outlined text-[16px]">sync</span>
                  Rotate
                </button>
                <button className="px-space-md py-1 rounded hover:bg-surface-container-high text-on-surface-variant hover:text-dark-text-primary text-body-sm font-body-sm flex items-center gap-1 transition-colors">
                  <span className="material-symbols-outlined text-[16px]">zoom_in</span>
                  Zoom
                </button>
              </div>
              <div className="flex items-center gap-space-md font-label-micro text-label-micro">
                <span className="text-dark-text-muted flex items-center gap-1">
                  LATENCY <span className="font-mono text-status-optimal font-bold">14ms</span>
                </span>
                <span className="text-dark-text-muted flex items-center gap-1">
                  DATA <span className="text-telemetry-cyan font-bold tracking-widest uppercase">Real-Time</span>
                </span>
              </div>
            </div>
          </div>

          {/* Lower Sub-Panels: 3-Column Engineering Matrix */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-space-md">
            
            {/* Panel 1: LLM Maintenance Priority */}
            <div className="bg-surface-container-low p-space-md rounded border border-dark-border-subtle flex flex-col justify-between shadow-sm">
              <div className="flex items-center justify-between pb-space-xs">
                <div className="flex flex-col">
                  <span className="font-label-micro text-label-micro uppercase text-dark-text-muted tracking-wider">AI Maintenance</span>
                  <span className="font-body-md text-body-md font-semibold text-dark-text-primary">
                    {llm?.maintenance_plan?.priority || 'Analyzing...'} Priority
                  </span>
                </div>
                <span className={`w-2 h-2 rounded-full ${llm?.maintenance_plan?.priority === 'High' ? 'bg-status-critical' : 'bg-status-warning'}`}></span>
              </div>
              <div className="py-2">
                <span className="font-label-micro text-label-micro uppercase text-dark-text-muted tracking-wider">Est. Timeframe</span>
                <div className="text-body-sm font-medium text-dark-text-primary mt-1">{llm?.maintenance_plan?.estimated_timeframe || '--'}</div>
              </div>
              <div className="flex items-center justify-between text-dark-text-muted font-label-micro text-label-micro pt-1 border-t border-dark-border-subtle mt-2">
                <span>LLM DIAGNOSTIC</span>
                <span className="text-primary hover:underline cursor-pointer">View Details</span>
              </div>
            </div>

            {/* Panel 2: LLM Actions */}
            <div className="bg-surface-container-low p-space-md rounded border border-dark-border-subtle flex flex-col shadow-sm col-span-2">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-dark-border-subtle">
                <span className="font-label-micro text-label-micro uppercase text-dark-text-muted tracking-wider">Recommended Actions</span>
                <div className="flex items-center gap-1 font-label-micro text-label-micro">
                  <span className="text-primary font-mono">Gemini 1.5</span>
                </div>
              </div>
              <ul className="flex flex-col gap-2 overflow-y-auto max-h-24 custom-scrollbar">
                {llm?.maintenance_plan?.recommended_actions?.map((act: string, i: number) => (
                  <li key={i} className="flex items-start gap-2 font-body-sm text-body-sm text-on-surface-variant">
                    <span className="material-symbols-outlined text-primary text-[16px] shrink-0 mt-0.5">chevron_right</span>
                    <span>{act}</span>
                  </li>
                )) || <div className="text-dark-text-muted text-sm">Waiting for LLM analysis...</div>}
              </ul>
            </div>
            
          </div>
        </div>

        {/* Right Master Column: Real-Time Insights & Operational Stream (4 Cols on Desktop) */}
        <div className="xl:col-span-4 flex flex-col gap-space-md">
          
          {/* Card 1: Primary Real-Time Insights */}
          <div className="bg-surface-container-low p-space-lg rounded border border-dark-border-subtle flex flex-col gap-space-md shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-space-sm">
                <span className="material-symbols-outlined text-primary text-[20px]">bolt</span>
                <h2 className="font-headline-md text-headline-md font-semibold text-dark-text-primary">Real-Time Insights</h2>
              </div>
            </div>
            
            {/* Bridge Operational State Highlight */}
            <div className="grid grid-cols-2 gap-space-md">
              <div className="bg-surface-container-high/60 border border-dark-border-subtle p-space-md rounded flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">Bridge Status</span>
                </div>
                <div className="mt-2 font-headline-lg text-headline-lg font-bold text-status-optimal flex items-center gap-1.5">
                  <span>Active</span>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-optimal opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-status-optimal"></span>
                  </span>
                </div>
                <span className="font-label-micro text-label-micro text-dark-text-muted mt-1">Both corridors open</span>
              </div>
              <div className="bg-surface-container-high/60 border border-dark-border-subtle p-space-md rounded flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">Main Span Stress</span>
                </div>
                <div className="mt-2 flex items-baseline gap-1.5">
                  <span className="font-headline-lg text-headline-lg font-bold text-status-optimal font-mono">
                    {telemetry?.Strain_microstrain?.toFixed(0) || '--'}
                  </span>
                  <span className="font-label-micro text-label-micro text-on-surface-variant font-mono">µε</span>
                </div>
                <span className="font-label-micro text-label-micro text-status-optimal mt-1">Optimal (&lt; 250 µε)</span>
              </div>
            </div>

            {/* Metric Boxes */}
            <div className="flex flex-col gap-space-xs">
              <div className="bg-surface-container-high/60 border border-dark-border-subtle p-space-md rounded flex items-center justify-between">
                <div className="flex items-center gap-space-md">
                  <div className="w-9 h-9 rounded bg-telemetry-blue/15 text-telemetry-blue flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">directions_car</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">Vehicle Load</span>
                    <span className="font-headline-md text-headline-md font-mono font-bold text-dark-text-primary">
                      {telemetry?.Traffic_Volume_vph?.toFixed(0) || '--'} <span className="font-body-sm text-body-sm font-normal text-on-surface-variant">vph</span>
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-surface-container-high/60 border border-dark-border-subtle p-space-md rounded flex items-center justify-between">
                <div className="flex items-center gap-space-md">
                  <div className="w-9 h-9 rounded bg-telemetry-cyan/15 text-telemetry-cyan flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">air</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">Wind Velocity</span>
                    <span className="font-headline-md text-headline-md font-mono font-bold text-dark-text-primary">
                      {telemetry?.Wind_Speed_ms?.toFixed(1) || '--'} <span className="font-body-sm text-body-sm font-normal text-on-surface-variant">m/s</span>
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="bg-surface-container-high/60 border border-dark-border-subtle p-space-md rounded flex items-center justify-between">
                <div className="flex items-center gap-space-md">
                  <div className="w-9 h-9 rounded bg-status-warning/15 text-status-warning flex items-center justify-center">
                    <span className="material-symbols-outlined text-[20px]">vibration</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="font-label-micro text-label-micro uppercase text-dark-text-muted">Vibration</span>
                    <span className="font-headline-md text-headline-md font-mono font-bold text-dark-text-primary">
                      {telemetry?.Vibration_ms2?.toFixed(2) || '--'} <span className="font-body-sm text-body-sm font-normal text-on-surface-variant">m/s²</span>
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Card 3: Explainable AI (XAI) Structural Degradation Drivers */}
          <div className="bg-surface-container-low p-space-lg rounded border border-dark-border-subtle flex flex-col gap-space-md shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-space-sm">
                <span className="material-symbols-outlined text-primary text-[20px]">psychology</span>
                <h3 className="font-headline-md text-headline-md font-semibold text-dark-text-primary">Degradation Drivers (XAI)</h3>
              </div>
              <span className="font-label-micro text-label-micro text-dark-text-muted font-mono">SHAP WEIGHTS</span>
            </div>
            
            <div className="flex flex-col gap-space-md pt-1">
              {xai?.engineering_factors?.slice(0, 4).map((eng: any, i: number) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between font-label-micro text-label-micro">
                    <span className="text-on-surface-variant">{eng.label}</span>
                    <span className="font-mono text-dark-text-primary font-semibold">{eng.weighted_share_percent}% Impact</span>
                  </div>
                  <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                    <div className="h-full bg-primary rounded-full" style={{ width: `${Math.min(100, eng.weighted_share_percent * 2)}%` }}></div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-1 p-space-sm rounded bg-surface-container-lowest border border-dark-border-subtle flex items-center justify-between">
              <span className="font-label-micro text-label-micro text-dark-text-muted">Predictive Model: LSTM-v4.1</span>
              <button className="font-label-micro text-label-micro text-primary hover:underline uppercase tracking-wider font-semibold">Deep Diagnostic</button>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
