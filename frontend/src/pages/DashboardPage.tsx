import { useState, useEffect, Suspense } from 'react';
import { 
  AlertTriangle, 
  CloudSun, 
  Gauge, 
  Cpu, 
  Wrench, 
  ShieldAlert, 
  Activity, 
  FileText, 
  Layers,
  Clock,
  Compass
} from 'lucide-react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, Grid, ContactShadows } from '@react-three/drei';
import { useBridgeContext } from '../context/BridgeContext';
import { apiClient } from '../api/client';

// ---------------------------------------------------------
// 3D Model Component with Dynamic Structural Heatmap
// ---------------------------------------------------------
const BridgeModelPlaceholder = ({ type, healthIndex }: { type: string; healthIndex?: number }) => {
  const health = healthIndex !== undefined ? healthIndex : 0.85;
  const deckColor = health > 0.75 ? '#3b82f6' : health > 0.5 ? '#f59e0b' : '#ef4444';
  const pillarColor = health > 0.75 ? '#64748b' : health > 0.5 ? '#94a3b8' : '#dc2626';

  return (
    <mesh position={[0, 1, 0]}>
      <boxGeometry args={[10, 0.5, 2]} />
      <meshStandardMaterial color={deckColor} metalness={0.6} roughness={0.3} />
      {/* Pillars */}
      <mesh position={[-3, -2, 0]}>
        <cylinderGeometry args={[0.3, 0.3, 4]} />
        <meshStandardMaterial color={pillarColor} />
      </mesh>
      <mesh position={[3, -2, 0]}>
        <cylinderGeometry args={[0.3, 0.3, 4]} />
        <meshStandardMaterial color={pillarColor} />
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
        setError(err.message || 'Failed to load bridge assessment.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchAssessment();
  }, [selectedBridgeId]);

  if (!selectedBridgeId) {
    return (
      <div className="p-gutter text-dark-text-muted flex items-center justify-center min-h-[50vh]">
        Please select a bridge from the header to view its digital twin.
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-gutter h-full flex items-center justify-center min-h-[50vh]">
        <div className="bg-error-container/20 border border-error-container/40 text-error p-4 rounded flex items-center gap-3">
          <AlertTriangle size={20} />
          {error}
        </div>
      </div>
    );
  }

  const bridge = assessment?.bridge;
  const fused = assessment?.fused_risk;
  const env = assessment?.environment;
  const weather = env?.weather;
  const traffic = env?.traffic;
  const shi = assessment?.structural_health;
  const telemetry = bridge?.telemetry?.latest;
  const llm = assessment?.llm_analysis;
  const plan = llm?.maintenance_plan;
  const xai = assessment?.explainability;
  const bridgeName = bridge?.name || 'Loading Bridge Asset...';

  // Risk priority color theme
  const priority = fused?.priority || 'Low';
  const priorityBadgeStyle =
    priority === 'Critical' ? 'bg-red-500/15 text-red-400 border-red-500/40'
    : priority === 'High' ? 'bg-orange-500/15 text-orange-400 border-orange-500/40'
    : priority === 'Medium' ? 'bg-amber-500/15 text-amber-400 border-amber-500/40'
    : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/40';

  const shiPercent = ((shi?.predicted_shi ?? 0.85) * 100).toFixed(1);
  const riskPercent = ((fused?.overall_risk_score ?? 0.15) * 100).toFixed(1);
  const confidencePercent = ((fused?.confidence_score ?? 0.85) * 100).toFixed(0);

  return (
    <div className="flex flex-col w-full p-gutter lg:p-gutter-lg gap-space-lg select-none text-on-surface pb-16">
      
      {/* ---------------------------------------------------------
          Top Executive Header & Key Health Bar
          --------------------------------------------------------- */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-space-md border-b border-dark-border-subtle pb-space-md">
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-space-sm">
            <h1 className="font-headline-lg text-headline-lg font-bold tracking-tight text-dark-text-primary">
              {bridgeName}
            </h1>
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded border text-xs font-semibold ${priorityBadgeStyle}`}>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-current"></span>
              </span>
              Priority: {priority}
            </span>
          </div>

          {/* Bridge Asset Specifications Strip */}
          <div className="flex flex-wrap items-center gap-space-md text-dark-text-muted text-xs">
            <span className="font-mono text-dark-text-primary font-medium">
              Asset ID: #{bridge?.id || selectedBridgeId}
            </span>
            <span>•</span>
            <span>Type: <strong className="text-dark-text-primary font-medium capitalize">{bridge?.structure_type || 'Girder'}</strong></span>
            <span>•</span>
            <span>Material: <strong className="text-dark-text-primary font-medium">{bridge?.material_type || 'Concrete / Steel'}</strong></span>
            <span>•</span>
            <span>Age: <strong className="text-dark-text-primary font-medium">{bridge?.age_years || 15} yrs</strong></span>
            <span>•</span>
            <span>Span: <strong className="text-dark-text-primary font-medium">{bridge?.length_m || 3800} m</strong></span>
            <span>•</span>
            <span className="text-cyan-400 font-mono text-[11px] uppercase tracking-wider">{bridge?.exposure_class || 'Severe Marine'}</span>
          </div>
        </div>

        {/* Right Header KPIs: Structural Health Index & Fused Risk */}
        <div className="flex flex-wrap items-center gap-space-sm">
          {/* Fused Risk Metric */}
          <div className="bg-surface-container-low px-space-md py-space-xs rounded border border-dark-border-subtle shadow-sm flex items-center gap-3">
            <Gauge className="w-5 h-5 text-indigo-400" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-dark-text-muted tracking-wider">Overall Fused Risk</span>
              <span className="text-sm font-mono font-bold text-dark-text-primary">
                {loading ? '--' : `${riskPercent}%`}
                <span className="text-xs text-dark-text-muted font-normal ml-1">({confidencePercent}% Conf)</span>
              </span>
            </div>
          </div>

          {/* Structural Health Index (SHI) Metric */}
          <div className="bg-surface-container-low px-space-md py-space-xs rounded border border-dark-border-subtle shadow-sm flex items-center gap-3">
            <Activity className="w-5 h-5 text-emerald-400" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-dark-text-muted tracking-wider">Structural Health Index</span>
              <span className="text-sm font-mono font-bold text-emerald-400">
                {loading ? '--' : `${shiPercent}%`}
                <span className="text-xs text-dark-text-muted font-normal ml-1">
                  ±{((shi?.mc_std || 0.02) * 100).toFixed(1)}%
                </span>
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ---------------------------------------------------------
          Live Environmental & Sensor Telemetry HUD Strip
          --------------------------------------------------------- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-3 rounded-lg bg-surface-container-lowest border border-dark-border-subtle">
        {/* Weather Tile */}
        <div className="flex items-center justify-between px-3.5 py-2.5 rounded bg-surface-container-low/70 border border-dark-border-subtle">
          <div className="flex items-center gap-3">
            <CloudSun className="w-5 h-5 text-amber-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-dark-text-muted tracking-wider">Live Ambient Weather</span>
              <span className="text-xs font-mono font-semibold text-dark-text-primary">
                {weather?.temperature_c || 30}°C • {weather?.humidity_percent || 65}% RH • {weather?.wind_speed_ms || 4} m/s
              </span>
            </div>
          </div>
          <span className={`text-[9px] px-2 py-0.5 rounded font-mono font-bold uppercase tracking-wider ${
            weather?.is_live ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' : 'bg-slate-700/40 text-slate-400'
          }`}>
            {weather?.is_live ? '● Open-Meteo' : 'Climatological'}
          </span>
        </div>

        {/* Traffic Tile */}
        <div className="flex items-center justify-between px-3.5 py-2.5 rounded bg-surface-container-low/70 border border-dark-border-subtle">
          <div className="flex items-center gap-3">
            <Compass className="w-5 h-5 text-cyan-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-dark-text-muted tracking-wider">Dynamic Traffic Flow</span>
              <span className="text-xs font-mono font-semibold text-dark-text-primary">
                {traffic?.congestion_level || 'Normal'} ({traffic?.current_vph?.toLocaleString() || '2,500'} vph) • {traffic?.load_multiplier || 1.0}x Load
              </span>
            </div>
          </div>
          <span className="text-[9px] px-2 py-0.5 rounded font-mono uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            {traffic?.is_live ? '● TomTom Flow' : 'Mumbai Diurnal'}
          </span>
        </div>

        {/* Structural Telemetry Tile */}
        <div className="flex items-center justify-between px-3.5 py-2.5 rounded bg-surface-container-low/70 border border-dark-border-subtle">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-indigo-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-semibold text-dark-text-muted tracking-wider">SHM Sensor Telemetry</span>
              <span className="text-xs font-mono font-semibold text-dark-text-primary">
                Strain: {telemetry?.Strain_microstrain ? `${telemetry.Strain_microstrain.toFixed(0)} µε` : '185 µε'} • Vib: {telemetry?.Vibration_ms2 ? `${telemetry.Vibration_ms2.toFixed(2)} m/s²` : '0.45 m/s²'}
              </span>
            </div>
          </div>
          <span className="text-[9px] px-2 py-0.5 rounded font-mono uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/30">
            Simulated SHM
          </span>
        </div>
      </div>

      {/* ---------------------------------------------------------
          Main 12-Column Responsive Dashboard Chassis
          --------------------------------------------------------- */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-lg items-start">
        
        {/* Left Master Column: 3D Twin & Detailed Gemini Directives (8 Cols) */}
        <div className="xl:col-span-8 flex flex-col gap-space-lg">
          
          {/* Digital Twin 3D Viewport Chassis */}
          <div className="relative w-full h-[470px] lg:h-[500px] rounded-lg bg-surface-container-lowest border border-dark-border-subtle overflow-hidden flex flex-col justify-between shadow-md">
            <div className="absolute inset-0 z-0">
               <Suspense fallback={<div className="w-full h-full flex items-center justify-center text-dark-text-muted text-sm">Loading 3D Twin...</div>}>
                  <Canvas camera={{ position: [5, 5, 10], fov: 45 }}>
                    <color attach="background" args={['#0e0e13']} />
                    <ambientLight intensity={0.5} />
                    <directionalLight position={[10, 10, 5]} intensity={1.5} color="#ffffff" castShadow />
                    <Environment preset="city" />
                    <BridgeModelPlaceholder 
                      type={bridge?.structure_type || 'girder'} 
                      healthIndex={fused?.overall_health_index || shi?.predicted_shi}
                    />
                    <Grid infiniteGrid fadeDistance={40} fadeStrength={5} cellColor="#333333" sectionColor="#1a1a1a" />
                    <ContactShadows position={[0, -2.5, 0]} opacity={0.6} scale={20} blur={2} far={4} color="#000000" />
                    <OrbitControls makeDefault minPolarAngle={0} maxPolarAngle={Math.PI / 2 - 0.1} />
                  </Canvas>
                </Suspense>
            </div>
            
            {/* Viewport Top Overlay: Spatial Controls */}
            <div className="relative z-10 p-space-md flex flex-wrap items-center justify-between gap-space-sm pointer-events-none">
              <div className="flex items-center gap-space-xs pointer-events-auto">
                <div className="px-space-md py-1.5 rounded bg-surface-container-highest/90 backdrop-blur-md text-dark-text-primary text-xs font-semibold flex items-center gap-1.5 shadow-sm border border-dark-border-subtle">
                  <Layers className="w-4 h-4 text-primary" />
                  <span>Interactive 3D Digital Twin ({bridge?.structure_type || 'Girder'})</span>
                </div>
              </div>
              
              {/* Multimodal Layer Controls */}
              <div className="flex items-center bg-surface-container-lowest/85 backdrop-blur-md p-1 rounded gap-1 shadow-md border border-dark-border-subtle pointer-events-auto">
                <button 
                  onClick={() => setActiveLayer('stress')}
                  className={`px-space-md py-1 rounded text-[11px] font-mono uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'stress' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full bg-accent-traffic ${activeLayer === 'stress' ? 'shadow-[0_0_8px_rgba(249,115,22,0.8)]' : ''}`}></span>
                  Stress Heatmap
                </button>
                <button 
                  onClick={() => setActiveLayer('traffic')}
                  className={`px-space-md py-1 rounded text-[11px] font-mono uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'traffic' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full bg-telemetry-cyan ${activeLayer === 'traffic' ? 'shadow-[0_0_8px_rgba(6,182,212,0.8)]' : ''}`}></span>
                  Traffic Dynamic
                </button>
                <button 
                  onClick={() => setActiveLayer('weather')}
                  className={`px-space-md py-1 rounded text-[11px] font-mono uppercase tracking-wider flex items-center gap-1.5 transition-all ${
                    activeLayer === 'weather' ? 'bg-surface-container-highest text-dark-text-primary' : 'text-on-surface-variant hover:text-dark-text-primary hover:bg-surface-container'
                  }`}
                >
                  <span className="w-2 h-2 rounded-full bg-outline"></span>
                  Marine Exposure
                </button>
              </div>
            </div>

            {/* Viewport Bottom HUD Control Bar */}
            <div className="relative z-10 p-space-sm mx-space-md mb-space-md rounded bg-surface-container-lowest/90 backdrop-blur-md flex flex-wrap items-center justify-between gap-space-md shadow-xl border border-dark-border-subtle pointer-events-auto">
              <div className="flex items-center gap-2 text-xs text-dark-text-muted">
                <span>Color-Coded Structural Health:</span>
                <span className="font-mono font-bold text-primary">{((fused?.overall_health_index || 0.85) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex items-center gap-space-md text-[11px] font-mono">
                <span className="text-dark-text-muted flex items-center gap-1">
                  LATENCY <span className="text-status-optimal font-bold">12ms</span>
                </span>
                <span className="text-dark-text-muted flex items-center gap-1">
                  TWIN <span className="text-telemetry-cyan font-bold uppercase">Active Mesh Sync</span>
                </span>
              </div>
            </div>
          </div>

          {/* ---------------------------------------------------------
              Detailed Gemini Predictive Maintenance Directive
              --------------------------------------------------------- */}
          <div className="bg-surface-container-low p-space-lg rounded-lg border border-dark-border-subtle shadow-sm flex flex-col gap-space-md">
            
            {/* Directive Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-dark-border-subtle pb-space-sm">
              <div className="flex items-center gap-space-sm">
                <div className="w-8 h-8 rounded bg-primary/10 border border-primary/30 flex items-center justify-center">
                  <FileText className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <h2 className="font-headline-md text-headline-md font-bold text-dark-text-primary">
                    Predictive Maintenance & Engineering Directive
                  </h2>
                  <span className="text-xs text-dark-text-muted">
                    Synthesized by Gemini 2.5 Flash Structural Intelligence & Physics-Informed Indicators
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-surface-container-highest border border-dark-border-subtle text-xs font-mono text-dark-text-primary">
                  <Clock className="w-3.5 h-3.5 text-primary" />
                  {plan?.estimated_timeframe || '30 calendar days'}
                </span>
              </div>
            </div>

            {/* Executive Structural Health Diagnosis (Detailed Narrative) */}
            <div className="p-space-md rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col gap-1.5">
              <span className="text-[10px] uppercase font-bold text-dark-text-muted tracking-wider">
                Executive Structural Diagnosis
              </span>
              <p className="text-xs leading-relaxed text-on-surface-variant font-normal">
                {llm?.analysis_narrative || 'Detailed civil engineering diagnosis synthesizing structural health telemetry, environmental stressors, and predictive degradation rate...'}
              </p>
            </div>

            {/* Root-Cause & Degradation Mechanics */}
            {plan?.root_cause_analysis && (
              <div className="p-space-md rounded bg-surface-container-lowest/80 border border-dark-border-subtle flex flex-col gap-1.5">
                <span className="text-[10px] uppercase font-bold text-amber-400 tracking-wider flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Root-Cause & Degradation Dynamics
                </span>
                <p className="text-xs leading-relaxed text-dark-text-muted">
                  {plan.root_cause_analysis}
                </p>
              </div>
            )}

            {/* Prioritized Actionable Engineering Work Packages */}
            <div className="flex flex-col gap-2 pt-1">
              <span className="text-[11px] uppercase font-bold text-dark-text-primary tracking-wider flex items-center gap-1.5">
                <Wrench className="w-4 h-4 text-primary" />
                Prioritized Action Work Packages
              </span>
              <div className="grid grid-cols-1 gap-2">
                {plan?.recommended_actions?.map((act: string, idx: number) => (
                  <div 
                    key={idx} 
                    className="p-3 rounded bg-surface-container-lowest border border-dark-border-subtle flex items-start gap-3 hover:border-primary/40 transition-colors"
                  >
                    <div className="w-5 h-5 rounded bg-primary/20 text-primary font-mono text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {idx + 1}
                    </div>
                    <div className="flex-1 text-xs text-on-surface-variant leading-relaxed">
                      {act}
                    </div>
                  </div>
                )) || (
                  <div className="text-xs text-dark-text-muted">Synthesizing maintenance actions...</div>
                )}
              </div>
            </div>

            {/* 2-Column Sub-Grid: Inspection Protocol & Operational Safety Mitigations */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-space-md pt-2 border-t border-dark-border-subtle">
              
              {/* Prescribed Inspection Protocol */}
              <div className="p-space-md rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col gap-2">
                <span className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider flex items-center gap-1.5">
                  <Compass className="w-3.5 h-3.5" />
                  Prescribed Inspection Protocol (NDT)
                </span>
                <p className="text-xs text-dark-text-muted leading-relaxed">
                  {plan?.inspection_protocol || 'Perform ultrasonic pulse velocity (UPV) and dynamic strain logging across critical spans during peak hours.'}
                </p>
                {plan?.required_equipment && (
                  <div className="mt-1 pt-2 border-t border-dark-border-subtle/50 flex flex-col gap-1">
                    <span className="text-[9px] uppercase font-semibold text-dark-text-muted">Required Equipment:</span>
                    <div className="flex flex-wrap gap-1">
                      {plan.required_equipment.map((eq: string, i: number) => (
                        <span key={i} className="text-[10px] px-2 py-0.5 rounded bg-surface-container-highest text-dark-text-muted border border-dark-border-subtle font-mono">
                          {eq}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Operational Safety Constraints */}
              <div className="p-space-md rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col gap-2">
                <span className="text-[10px] uppercase font-bold text-orange-400 tracking-wider flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Operational Safety Mitigations
                </span>
                <ul className="flex flex-col gap-1.5">
                  {plan?.safety_mitigations?.map((sm: string, i: number) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-dark-text-muted leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-orange-400 shrink-0 mt-1.5"></span>
                      <span>{sm}</span>
                    </li>
                  )) || (
                    <li className="text-xs text-dark-text-muted">Maintain standard operational parameters and monitor telemetry alerts.</li>
                  )}
                </ul>
              </div>

            </div>

          </div>

        </div>

        {/* Right Master Column: Unified Health, Risk Fusion & Degradation Drivers (4 Cols) */}
        <div className="xl:col-span-4 flex flex-col gap-space-lg">
          
          {/* Card 1: Unified Structural Health & Risk Fusion */}
          <div className="bg-surface-container-low p-space-lg rounded-lg border border-dark-border-subtle flex flex-col gap-space-md shadow-sm">
            <div className="flex items-center justify-between border-b border-dark-border-subtle pb-space-xs">
              <div className="flex items-center gap-space-sm">
                <Gauge className="w-5 h-5 text-indigo-400" />
                <h3 className="font-headline-md text-headline-md font-bold text-dark-text-primary">
                  Structural Risk & Health
                </h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                FUSED ENGINE
              </span>
            </div>

            {/* Dominant Stressor Header */}
            <div className="p-3 rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col gap-1">
              <span className="text-[10px] uppercase font-bold text-dark-text-muted tracking-wider">
                Dominant Structural Stressor
              </span>
              <span className="text-xs font-semibold text-dark-text-primary">
                {fused?.dominant_stressor || 'Marine Environmental Degradation & Cyclic Load'}
              </span>
            </div>

            {/* 3-Way Risk Fusion Breakdown Progress Bars */}
            <div className="flex flex-col gap-3 pt-1">
              <span className="text-[10px] text-dark-text-muted uppercase tracking-wider font-bold">
                Weighted Risk Components
              </span>
              
              {/* Machine Learning Propensity */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-xs">
                  <span className="text-on-surface-variant">Machine Learning Risk Propensity</span>
                  <span className="font-mono text-dark-text-primary font-semibold">
                    35% w • {((fused?.sub_scores?.ml_risk || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-indigo-500 rounded-full" 
                    style={{ width: `${Math.min(100, (fused?.sub_scores?.ml_risk || 0) * 100)}%` }}
                  ></div>
                </div>
              </div>

              {/* Sensor Telemetry Risk */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-xs">
                  <span className="text-on-surface-variant">Sensor Telemetry Risk (1 - SHI)</span>
                  <span className="font-mono text-dark-text-primary font-semibold">
                    35% w • {((fused?.sub_scores?.health_risk || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-cyan-500 rounded-full" 
                    style={{ width: `${Math.min(100, (fused?.sub_scores?.health_risk || 0) * 100)}%` }}
                  ></div>
                </div>
              </div>

              {/* Physics Degradation Risk */}
              <div className="flex flex-col gap-1">
                <div className="flex justify-between text-xs">
                  <span className="text-on-surface-variant">Physics Degradation (Corrosion/Fatigue)</span>
                  <span className="font-mono text-dark-text-primary font-semibold">
                    30% w • {((fused?.sub_scores?.engineering_risk || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-amber-500 rounded-full" 
                    style={{ width: `${Math.min(100, (fused?.sub_scores?.engineering_risk || 0) * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Asset Degradation Velocity Box */}
            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-dark-border-subtle">
              <div className="p-2.5 rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col">
                <span className="text-[10px] text-dark-text-muted uppercase font-mono">Annual Loss Rate</span>
                <span className="text-sm font-bold font-mono text-dark-text-primary mt-1">
                  {((assessment?.engineering_indicators?.annual_degradation_rate || 0.005) * 100).toFixed(2)}% / yr
                </span>
              </div>
              <div className="p-2.5 rounded bg-surface-container-lowest border border-dark-border-subtle flex flex-col">
                <span className="text-[10px] text-dark-text-muted uppercase font-mono">Fatigue Index</span>
                <span className="text-sm font-bold font-mono text-dark-text-primary mt-1">
                  {((assessment?.engineering_indicators?.fatigue_damage_fraction || 0.02) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          {/* Card 2: Structural Degradation Drivers (Explainable AI - XAI) */}
          <div className="bg-surface-container-low p-space-lg rounded-lg border border-dark-border-subtle flex flex-col gap-space-md shadow-sm">
            <div className="flex items-center justify-between border-b border-dark-border-subtle pb-space-xs">
              <div className="flex items-center gap-space-sm">
                <Compass className="w-5 h-5 text-primary" />
                <h3 className="font-headline-md text-headline-md font-bold text-dark-text-primary">
                  Degradation Drivers (XAI)
                </h3>
              </div>
              <span className="text-[10px] text-dark-text-muted font-mono">
                PHYSICS ATTRIBUTION
              </span>
            </div>
            
            <div className="flex flex-col gap-3 pt-1">
              {xai?.engineering_factors?.slice(0, 4).map((eng: any, i: number) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-on-surface-variant">{eng.label}</span>
                    <span className="font-mono text-dark-text-primary font-semibold">{eng.weighted_share_percent}% Impact</span>
                  </div>
                  <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary rounded-full" 
                      style={{ width: `${Math.min(100, eng.weighted_share_percent * 2)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-1 p-space-sm rounded bg-surface-container-lowest border border-dark-border-subtle flex items-center justify-between text-xs">
              <span className="text-dark-text-muted text-[11px]">Engineering Models:</span>
              <span className="text-primary font-mono font-semibold text-[11px]">Basquin S-N / Euler-Bernoulli</span>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
};

export default DashboardPage;
