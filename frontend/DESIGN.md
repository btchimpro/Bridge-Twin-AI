# BridgeTwin AI - Dashboard Design Specification

## Overview
BridgeTwin is an engineering intelligence platform for monitoring the structural health of bridge infrastructure (Digital Twin). The aesthetic is a premium, ultra-minimalist dark mode (inspired by Vercel/Stripe). It features deep black backgrounds, delicate monochromatic borders, and precise typography.

## Global Design System
- **Theme**: Dark Mode only.
- **Backgrounds**: 
  - Main Body: True Black (`#000000`)
  - Sidebar / Header: Flat Slate (`#0a0a0a`)
  - Component Cards: Matte Dark Gray (`#111111`)
- **Borders**: Extremely subtle (`rgba(255, 255, 255, 0.05)` or `#222222`). No heavy lines.
- **Typography**: `Inter` font. High contrast. 
  - Primary text: Crisp Off-White (`#ededed`)
  - Secondary text / Labels: Muted Gray (`#a1a1aa`). Small caps, wide tracking for section headers.
- **Accents**: No large glowing backgrounds. Use tiny, sharp dot indicators (Emerald Green `#10b981` for Healthy, Rose Red `#f43f5e` for High Risk, Amber `#f59e0b` for Warnings).
- **Shadows**: None. Completely flat, 2D UI.

## Application Layout
A full-screen web application divided into two main areas:
1. **Left Sidebar (256px wide)**:
   - Header with Logo (Square icon) and "BridgeTwin" title.
   - Navigation links (Dashboard, GIS Overview). Active state has a subtle `bg-white/10` background.
   - Bottom pinned "Settings" link.
2. **Main Content Area (Flex-1)**:
   - **Top Header (56px high)**: Contains a Bridge Selector Dropdown on the left, and notification bell/user avatar on the right.
   - **Dashboard Page**: Scrollable content area.

## Dashboard Page Structure
The Dashboard is composed of a responsive grid (1 column on mobile, 3 columns on large screens `xl:grid-cols-3`).

### 1. Page Header (Full Width)
- **Left**: Bridge Name (`<h2>`) and a tiny pulsing "Live Monitoring Active" indicator.
- **Right**: Two metric cards side-by-side.
  - Card 1: "LSTM SHI" (Structural Health Index). Large number out of 100 (e.g., `89.5 / 100`).
  - Card 2: "XGBoost Risk". Text showing "Low Risk" or "High Risk" with a 2px left border (Emerald or Rose).

### 2. Left Column (Spans 2 columns on Desktop)
- **3D Visualization Panel**: 
  - A large dark container (min-height 400px) representing a 3D canvas.
  - Top-left floating badge: "Layer: Structure".
- **Maintenance Timeline (Bottom)**: 
  - Header: "Maintenance Timeline" with a wrench icon.
  - Split layout: Left side shows "Priority" (e.g., HIGH) and "Timeframe" (e.g., 30 Days). Right side shows a paragraph of AI analysis and a bulleted list of 2-3 recommended actions.

### 3. Right Column (Spans 1 column on Desktop)
- **Live Edge Telemetry Panel**:
  - Header: "Live Edge Telemetry" with a tiny "Simulated" badge.
  - A 2x2 tight grid of sensor metrics (Traffic Vol, Wind Speed, Strain, Vibration). Each cell has a tiny icon, an uppercase label, and a large white number.
- **Explainability (XAI) Panel**:
  - Header: "Explainability (XAI)".
  - Section 1: "Degradation Drivers". A list of 4 factors (e.g., Corrosion, Fatigue). Each has a label, a percentage, and a very thin (2px) progress bar line.
  - Section 2: "LSTM Stressors". A list of 3 telemetry factors driving the health down, showing the metric value and a red/green arrow with a percentage impact.
