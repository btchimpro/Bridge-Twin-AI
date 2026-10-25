import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { apiClient } from '../api/client';

export interface BridgeInfo {
  id: string;
  name: string;
  lat: number;
  lon: number;
  status: string;
  risk: string;
  priority?: string;
  consensus?: string;
  ensemble_risk_probability?: number;
}

interface BridgeContextType {
  bridges: BridgeInfo[];
  selectedBridgeId: string | null;
  setSelectedBridgeId: (id: string) => void;
  loading: boolean;
  error: string | null;
}

const BridgeContext = createContext<BridgeContextType | undefined>(undefined);

export function BridgeProvider({ children }: { children: ReactNode }) {
  const [bridges, setBridges] = useState<BridgeInfo[]>([]);
  const [selectedBridgeId, setSelectedBridgeId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchBridges = async () => {
      try {
        const { data } = await apiClient.get<BridgeInfo[]>('/api/bridges');
        setBridges(data);
        if (data.length > 0) {
          setSelectedBridgeId(data[0].id);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load bridges');
      } finally {
        setLoading(false);
      }
    };
    
    fetchBridges();
  }, []);

  return (
    <BridgeContext.Provider value={{ bridges, selectedBridgeId, setSelectedBridgeId, loading, error }}>
      {children}
    </BridgeContext.Provider>
  );
}

export function useBridgeContext() {
  const context = useContext(BridgeContext);
  if (context === undefined) {
    throw new Error('useBridgeContext must be used within a BridgeProvider');
  }
  return context;
}
