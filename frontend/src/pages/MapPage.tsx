import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { useEffect } from 'react';
import { MapPin } from 'lucide-react';
import { useBridgeContext } from '../context/BridgeContext';

// Fix for leaflet markers in React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Component to recenter map when bridge changes
const MapRecenter = ({ lat, lon }: { lat: number, lon: number }) => {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lon], map.getZoom(), { animate: true });
  }, [lat, lon, map]);
  return null;
};

const MapPage = () => {
  const { bridges, selectedBridgeId, setSelectedBridgeId, loading } = useBridgeContext();

  if (loading) return <div className="p-6">Loading GIS data...</div>;

  const activeBridge = bridges.find(b => b.id === selectedBridgeId) || bridges[0];

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-semibold text-white">GIS Infrastructure Overview</h2>
          <p className="text-slate-400 mt-1">Mumbai Metropolitan Region - AI Monitoring Active</p>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 flex-1 min-h-0">
        <div className="lg:w-1/3 flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar">
          {bridges.map(bridge => (
            <div 
              key={bridge.id}
              onClick={() => setSelectedBridgeId(bridge.id)}
              className={`glass-panel p-4 rounded-xl cursor-pointer transition-all border ${
                activeBridge.id === bridge.id ? 'border-indigo-500/50 bg-indigo-500/10' : 'border-white/5 hover:bg-slate-800/40'
              }`}
            >
              <div className="flex justify-between items-start">
                <h3 className="font-medium text-white flex items-center gap-2">
                  <MapPin size={16} className={bridge.status === 'Warning' ? 'text-amber-500' : 'text-emerald-500'} />
                  {bridge.name}
                </h3>
                <span className={`text-xs px-2 py-1 rounded-full ${
                  bridge.status === 'Warning' ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/20 text-emerald-400'
                }`}>
                  {bridge.status}
                </span>
              </div>
              
              <div className="mt-4 grid grid-cols-2 gap-2">
                <div className="bg-slate-900/50 p-2 rounded flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider">AI Risk Model</span>
                  <span className="text-sm text-slate-300 font-medium">{bridge.risk}</span>
                </div>
                <div className="bg-slate-900/50 p-2 rounded flex flex-col">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider">Sensors</span>
                  <span className="text-sm text-slate-300 font-medium">Online</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex-1 glass-panel rounded-xl overflow-hidden relative border border-white/10 z-0">
          {activeBridge && (
            <MapContainer 
              center={[activeBridge.lat, activeBridge.lon]} 
              zoom={11} 
              className="w-full h-full"
              style={{ background: '#0f172a' }}
            >
              <MapRecenter lat={activeBridge.lat} lon={activeBridge.lon} />
              <TileLayer
                attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
              />
              {bridges.map(bridge => (
                <Marker 
                  key={bridge.id} 
                  position={[bridge.lat, bridge.lon]}
                  eventHandlers={{
                    click: () => setSelectedBridgeId(bridge.id),
                  }}
                >
                  <Popup className="custom-popup">
                    <div className="font-semibold">{bridge.name}</div>
                    <div className="text-xs mt-1">Status: {bridge.status}</div>
                  </Popup>
                </Marker>
              ))}
            </MapContainer>
          )}
        </div>
      </div>
    </div>
  );
};

export default MapPage;
