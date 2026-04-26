FairChain - Product Requirements Document
1. Executive Summary
FairChain is an advanced, real-time supply chain intelligence platform designed to ingest multi-source transit data, predict logistical disruptions before they cascade using sophisticated machine learning anomaly detection and time-series forecasting algorithms, and automatically generate optimized rerouting recommendations. This core functionality is enhanced by a powerful Open Innovation layer: a Supplier Fairness Auditor that continuously monitors AI-driven supply chain decisions for systemic bias against demographic groups such as small vendors, women-owned businesses, or developing-economy suppliers. The entire decision-making process is rendered transparent via a Google Gemini-powered explainability layer that translates complex probabilistic ML outputs into clear, actionable natural language alerts.
Targeting the PromptWars Solution Challenge 2026, this 4-5 page PRD outlines the ambitious technical scope, data strategy, and architectural decisions guiding a 4-person engineering team.
2. Comprehensive Problem Statement: The Indian Supply Chain Crisis
2.1 The Reactive Nature of Current Logistics
The Indian logistics ecosystem currently manages millions of shipments daily across a highly fragmented, interconnected network of highways, rail corridors, and maritime ports. Key disruption sources are relentless and multifaceted:
Seasonal severe weather events: e.g., the recurring Kerala floods, Odisha cyclones, and chronic Mumbai monmonsoons.
Infrastructure bottlenecks: Road closures and localized highway blockages frequently reported by NHAI.
Maritime congestion: Port pileups at JNPort, Mundra, and Chennai causing massive container throughput delays.
Socio-economic friction: Fuel strikes, spontaneous blockades, and acute driver shortages.
Crucially, a single localized bottleneck invariably triggers 3 to 5 downstream shipment failures. Logistics operators currently identify disruptions strictly post-factumùonly after delivery timelines are explicitly missed. There is no predictive shield.
2.2 The Open Innovation Complement: Hidden Bias in Supplier Allocation
As supply chains digitize, AI models increasingly dictate which vendors are awarded contracts, which routes are prioritized, and which suppliers are dropped for underperformance. However, these AI models optimize exclusively for mathematical efficiency, inheriting stark historical prejudices. A small supplier located in a historically flood-prone region might receive a low AI 'trust score', severely crippling their business prospects, even if their isolated performance matches that of massive enterprise conglomerates. FairChain rectifies this by layering fairness metrics natively into the logistics intelligence suite.
3. Deep Technical Solution Architecture
3.1 Preemptive Detection Engine (Core Functionality)
The central thesis of FairChain is preemptive detection. Extensive historical analysis of Indian freight network disruptions reveals consistent pre-disruption signaturesùvelocity drops, extreme route variance spikes, and specific weather pattern correlationsùthat manifest 4 to 8 hours before hard delays materialize. FairChain's ML pipelines ingest these signatures via an Isolation Forest anomaly detection model and an LSTM/Prophet time-series forecasting cluster to flag these vulnerabilities preemptively.
3.2 Dynamic Network Routing (Core Functionality)
Once a disruption is mathematically highly probable (> 75% confidence), the system engages a NetworkX directional graph representation of the entire Indian National Highway network. It dynamically recalculates edge weights based on real-time risk scores and triggers Yen's k-shortest paths algorithm to surface the top three viable alternative trajectories, balancing temporal cost, distance, and secondary risk parameters.
3.3 System Architecture Data Flow
The system architecture relies on asynchronous, real-time data streaming:
[External APIs: IMD Weather, OSM Road Network, Simulated Freight Telematics] -> Ingested by FastAPI Pipeline.
[FastAPI Pipeline] -> Triggers ML Layer (Isolation Forest for Anomalies, Prophet for Delay Horizon, AIF360 for Bias Auditing).
[ML Layer] -> Pushes state changes to Supabase PostgreSQL database.
[Supabase PostgreSQL] -> Broadcasts state changes via Realtime WebSockets to Next.js Client.
[Next.js Client] -> Renders Mapbox GL JS animated trajectories, Risk Overlays, and LLM alerts.
4. Exhaustive Technical Stack
The robust technology stack selected for the FairChain MVP comprises:
Frontend: React.js (Next.js 14) leveraged for SSR performance and SEO, styled with Tailwind CSS and radix-ui/shadcn for rapid, accessible component composition.
Geospatial Visualization: Mapbox GL JS for fluid, GPU-accelerated rendering of complex Indian highway polygons and thousands of concurrent animated shipment vectors.
Backend Services: FastAPI (Python 3.10+) serving as the high-throughput, async-capable bridge between the frontend and the heavy ML computation layers.
Database and Real-Time Sync: Supabase (PostgreSQL 15). Chosen specifically for its native Row Level Security and powerful 'Realtime' WebSocket broadcasting capabilities for instant dashboard updates.
Disruption Machine Learning: Scikit-Learn (Isolation Forest) for rapid anomaly scoring; Prophet and statsmodels (ARIMA) for granular time-series horizon forecasting.
Routing Algorithms: NetworkX for robust graph-theoretical modeling of the subcontinent's freight arteries.
Bias and Fairness ML Layer: AI Fairness 360 (AIF360) and Microsoft Fairlearn for strict mathematical disparate impact analysis and equalized odds calculations.
Explainability Engine: Google Gemini API for producing high-quality, actionable natural language summarizations of complex multi-variate ML outputs.
5. Detailed Core Feature Specifications (MVP)
5.1 Feature 1: Live Route Intelligence Viewport
The primary dashboard interface provides a god's-eye view of active supply chain logistics.
Deep integration with OSM (OpenStreetMap) data filtered specifically for NH (National Highway) topologies.
Displays hundreds of simulated 'in-transit' commercial shipments mapped strictly to road geometries.
Implements a dynamic risk gradient UI (Green -> Yellow -> Orange -> Red) painted over specific road segments based on the live ML risk score.
5.2 Feature 2: Time-Series Anomaly Detection Matrix
The central predictive organ of the platform.
Continuously evaluates vectors containing shipment velocity, physical transit time variance, aggregate corridor delay frequency, and localized weather intensity (rainfall mm/hr, wind speed km/h).
Operates explicitly on the 4-8 hour lead-time window, allowing logistical managers actual actionable temporal headroom.
5.3 Feature 3: Automated Dynamic Rerouting Engine
Directly mitigates detected anomalies.
Calculates 'distance vs. time vs. risk' matrices for affected shipments.
Constructs user-friendly 'Accept/Reject' interface modals for logistical operators to instantly reroute active fleets with a single interaction.
5.4 Feature 4: Algorithmic Supplier Fairness Scorecard
The specific Open Innovation deliverable.
Maintains a live audit of the AI determining vendor reliability scores.
Quantifies statistical parity differences separating baseline logistics performance and the AI's 'trust score'.
Visualizes these discrepancies, demonstrating how minor structural impediments unfairly penalize minority/small enterprise vendors.
5.5 Feature 5: LLM (Gemini) Explanation Console
Bridges the gap between raw statistical ML output and human operations.
Translates a 6-variable Isolation Forest output array into: 'High confidence delay on NH48. Sudden 45mm/hr rainfall spike combined with 12% fleet velocity drop indicates imminent severe flooding.'
6. Data Curation Strategy and Simulation Mechanics
Given the strict parameters of hackathon development, FairChain balances open data integration with sophisticated simulation modeling:
Historical Training Base: Models are trained against genuine records from severe events (e.g., Nov 2023 Chennai Floods, 2018 Kerala Floods), utilizing IMD weather archives correlated with NHAI closure logs.
Live Simulation Engine: To demonstrate the system's real-time capabilities to judges, a backend simulation loops over the timeline of a historical event, feeding precise historical telemetry to the ML engines as if occurring live, validating exactly when the model flags the event versus the actual chronological historical closure.
