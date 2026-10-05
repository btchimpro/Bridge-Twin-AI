import { Outlet, NavLink } from 'react-router-dom';
import { useBridgeContext } from '../context/BridgeContext';

const BridgeSelector = () => {
  const { bridges, selectedBridgeId, setSelectedBridgeId, loading } = useBridgeContext();

  if (loading) {
    return <div className="text-body-sm text-dark-text-muted flex items-center h-9">Loading bridges...</div>;
  }

  const selectedBridge = bridges.find(b => b.id === selectedBridgeId);

  return (
    <div className="relative group">
      <div className="flex items-center gap-space-xs bg-surface-container-lowest border border-dark-border-subtle rounded px-space-sm py-1 cursor-pointer hover:border-outline-variant transition-colors">
        <span className="material-symbols-outlined text-primary text-[16px]">domain</span>
        <span className="font-body-sm text-body-sm font-medium text-dark-text-primary whitespace-nowrap">
          {selectedBridge?.name || 'Select a bridge'}
        </span>
        <span className="material-symbols-outlined text-on-surface-variant text-[16px]">unfold_more</span>
      </div>
      
      {/* Dropdown Menu - appears on hover */}
      <div className="absolute top-full left-0 mt-1 w-full bg-surface-container-highest border border-dark-border-subtle rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 overflow-hidden">
        {bridges.map(bridge => (
          <button
            key={bridge.id}
            onClick={() => setSelectedBridgeId(bridge.id)}
            className={`w-full text-left px-space-md py-space-sm text-body-sm flex items-center justify-between hover:bg-surface-container transition-colors ${
              bridge.id === selectedBridgeId ? 'text-primary' : 'text-on-surface-variant'
            }`}
          >
            <span className="truncate">{bridge.name}</span>
            <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ml-2 ${bridge.risk === 'High' ? 'bg-status-critical' : 'bg-status-optimal'}`} />
          </button>
        ))}
      </div>
    </div>
  );
};

const MainLayout = () => {
  return (
    <div className="bg-dark-bg-base font-body-md text-on-surface antialiased min-h-screen">
      <aside className="fixed left-0 top-0 h-full w-64 bg-dark-bg-subtle border-r border-dark-border-subtle z-50 flex flex-col justify-between select-none">
        <div className="flex flex-col">
          <div className="h-14 px-gutter flex items-center justify-between border-b border-dark-border-subtle">
            <div className="flex items-center gap-space-sm">
              <div className="w-7 h-7 rounded bg-primary-container/20 border border-primary/40 flex items-center justify-center">
                <span className="material-symbols-outlined text-primary text-[18px]">polyline</span>
              </div>
              <span className="font-headline-md text-headline-md tracking-tight text-dark-text-primary font-semibold">BridgeTwin</span>
            </div>
          </div>
          
          <div className="px-gutter pt-space-lg pb-space-xs">
            <span className="font-label-micro text-label-micro uppercase tracking-wider text-dark-text-muted">Engineering Operations</span>
          </div>
          
          <nav className="flex flex-col gap-0.5 px-space-xs">
            <NavLink
              to="/dashboard"
              className={({ isActive }) =>
                `flex items-center justify-between px-space-md py-space-sm rounded transition-colors ${
                  isActive
                    ? 'bg-surface-container-highest text-dark-text-primary font-medium border-l-2 border-primary'
                    : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface border-l-2 border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-space-md">
                <span className="material-symbols-outlined text-[18px]">space_dashboard</span>
                <span className="font-body-md text-body-md">Overview</span>
              </div>
            </NavLink>
            
            <NavLink
              to="/map"
              className={({ isActive }) =>
                `flex items-center justify-between px-space-md py-space-sm rounded transition-colors ${
                  isActive
                    ? 'bg-surface-container-highest text-dark-text-primary font-medium border-l-2 border-primary'
                    : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface border-l-2 border-transparent'
                }`
              }
            >
              <div className="flex items-center gap-space-md">
                <span className="material-symbols-outlined text-[18px]">map</span>
                <span className="font-body-md text-body-md">GIS Overview</span>
              </div>
            </NavLink>
          </nav>
        </div>
        
        <div className="p-space-xs border-t border-dark-border-subtle flex flex-col gap-1">
          <div className="px-space-md py-space-xs flex items-center justify-between bg-surface-container-lowest/60 rounded border border-dark-border-subtle">
            <div className="flex items-center gap-space-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-status-optimal opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-status-optimal"></span>
              </span>
              <span className="font-label-micro text-label-micro text-on-surface-variant uppercase tracking-wider">All systems nominal</span>
            </div>
            <span className="font-label-micro text-label-micro text-status-optimal font-mono">0ms</span>
          </div>
          <button className="flex items-center gap-space-md px-space-md py-space-sm rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors w-full">
            <span className="material-symbols-outlined text-[18px]">settings</span>
            <span className="font-body-md text-body-md">Platform Settings</span>
          </button>
        </div>
      </aside>
      
      <div className="pl-64">
        <header className="fixed top-0 left-64 right-0 h-14 bg-dark-bg-subtle/95 backdrop-blur-md border-b border-dark-border-subtle z-40 flex items-center justify-between px-gutter">
          <div className="flex items-center gap-space-lg flex-1 max-w-2xl">
            <BridgeSelector />
            <div className="relative flex-1">
              <span className="material-symbols-outlined absolute left-space-sm top-1/2 -translate-y-1/2 text-[16px] text-dark-text-muted">search</span>
              <input 
                className="w-full bg-surface-container-lowest border border-dark-border-subtle rounded pl-7 pr-space-md py-1 font-body-sm text-body-sm text-dark-text-primary placeholder:text-dark-text-muted focus:outline-none focus:border-primary/50 transition-colors" 
                placeholder="Search telemetry, sensors, alerts..." 
                type="text"
              />
            </div>
          </div>
          
          <div className="flex items-center gap-space-md">
            <div className="hidden md:flex items-center gap-space-xs font-label-micro text-label-micro text-dark-text-muted border-r border-dark-border-subtle pr-space-md">
              <span className="material-symbols-outlined text-[14px]">schedule</span>
              <span className="font-mono">Live</span>
            </div>
            <button className="w-8 h-8 rounded bg-surface-container-lowest border border-dark-border-subtle flex items-center justify-center text-on-surface-variant hover:text-dark-text-primary hover:border-outline-variant transition-colors" type="button">
              <span className="material-symbols-outlined text-[18px]">filter_list</span>
            </button>
            <button className="relative w-8 h-8 rounded bg-surface-container-lowest border border-dark-border-subtle flex items-center justify-center text-on-surface-variant hover:text-dark-text-primary hover:border-outline-variant transition-colors" type="button">
              <span className="material-symbols-outlined text-[18px]">notifications</span>
              <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-error"></span>
            </button>
            <div className="flex items-center gap-space-sm pl-space-xs">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                <span className="material-symbols-outlined text-on-primary text-[18px]">person</span>
              </div>
              <div className="hidden 2xl:flex flex-col text-left">
                <span className="font-body-sm text-body-sm font-semibold text-dark-text-primary leading-none">K. Rampelli</span>
                <span className="font-label-micro text-label-micro text-dark-text-muted mt-0.5">Structural Lead</span>
              </div>
            </div>
          </div>
        </header>
        
        <main className="relative pt-14 bg-dark-bg-base min-h-screen">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
