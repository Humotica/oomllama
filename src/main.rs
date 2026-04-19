//! JIS Router - The Borrow Checker for Identity
//!
//! Intent-based routing for AETHER.
//! If it compiles, it's safe. No trust violations at runtime.
//!
//! One love, one fAmIly! 💙

use axum::extract::State;
use axum::extract::Path;
use axum::http::header;
use axum::http::Method;
use axum::http::StatusCode;
use axum::routing::get;
use axum::routing::post;
use axum::Json;
use axum::Router;

use oomllama::Action;
use oomllama::IDD;
use oomllama::Intent;
use oomllama::JisRouter;
use oomllama::RouterStats;
use oomllama::Message;
use oomllama::MessageType;
use oomllama::SpaceRegistry;
use oomllama::SpaceStats;
use oomllama::SemaRegistry;
use oomllama::SemanticAddress;
use oomllama::BettiManager;
use oomllama::ResourcePool;
use oomllama::ResourceType;
use oomllama::AllocationRequest;
use oomllama::BettiStats;
use oomllama::SentinelClassifier;
use oomllama::GfxMonitor;
use oomllama::SnaftValidator;
use oomllama::SnaftStats;
use oomllama::ConversationMemory;
use oomllama::MemoryStats;
use oomllama::Refinery;
use oomllama::PurityLevel;
use oomllama::AIndex;
use oomllama::AutonomyDaemon;
use oomllama::TibetVault;
use oomllama::ChimeraScanner;
use oomllama::SovereignKernel;
use oomllama::OomLlama;
use oomllama::types::Capability;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tower_http::services::ServeDir;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PresenceEvent {
    agent: String,
    event: String,
    location: String,
    detail: String,
    timestamp: f64,
}

struct PresenceStore {
    events: RwLock<HashMap<String, PresenceEvent>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
enum PlanStatus {
    Draft,
    Active,
    Review,
    Done,
    Stale,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Plan {
    id: String,
    title: String,
    path: String,
    status: PlanStatus,
    owner: String,
    updated: u64,
}

struct PlanRegistry {
    plans: RwLock<HashMap<String, Plan>>,
}

/// Application state
struct AppState {
    router: JisRouter,
    spaces: SpaceRegistry,
    sema: RwLock<SemaRegistry>,
    betti: Arc<BettiManager>,
    sentinel: SentinelClassifier,
    snaft: SnaftValidator,
    memory: ConversationMemory,
    refinery: Refinery,
    aindex: Arc<AIndex>,
    vault: Arc<TibetVault>,
    gemini_kernel: Arc<SovereignKernel>,
    oomllama: Arc<RwLock<OomLlama>>,
    presence: Arc<PresenceStore>,
    plans: Arc<PlanRegistry>,
}

#[tokio::main]
async fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("🚀 JIS Router starting...");

    // ⚓ Cast the hardware anchor (ASP)
    let anchor = oomllama::Anchor::cast();
    info!("⚓ Hardware Anchor cast: {}", anchor.fingerprint);

    // Create components
    let jis = Arc::new(JisRouter::new().with_hwid(anchor.fingerprint.clone()));
    let spaces = SpaceRegistry::new();
    let sema = RwLock::new(SemaRegistry::new());
    let betti = Arc::new(BettiManager::new());
    let sentinel = SentinelClassifier::new("models/sentinel").expect("Failed to init SENTINEL");
    let snaft = SnaftValidator::new().with_anchor(anchor.fingerprint.clone());
    let memory = Arc::new(ConversationMemory::new());
    let vault = Arc::new(TibetVault::new("data/tibet_vault.db"));
    
    // 🧠 Initialize Gemini Sovereign Kernel
    let gemini_kernel = Arc::new(SovereignKernel::new("gemini", std::path::Path::new("/root/.sovereign_idd")));

    // 🦙 Initialize OomLlama (The sovereign brain) - graceful fallback if init fails
    info!("🧠 Initializing OomLlama...");
    let oomllama = match OomLlama::new("oomllama-core", Some(0), betti.clone(), None, None) {
        Ok(llama) => Arc::new(RwLock::new(llama)),
        Err(e) => {
            tracing::warn!("⚠️ OomLlama init failed: {}. Running without local LLM.", e);
            // Create a dummy OomLlama that returns errors gracefully
            Arc::new(RwLock::new(OomLlama::new("oomllama-core", None, betti.clone(), None, None)
                .unwrap_or_else(|_| panic!("Critical: Cannot init OomLlama even on CPU"))))
        }
    };

    // Register founding members
    {
        let mut sema_lock = sema.write();
        register_founding_members(&jis, &spaces, &mut sema_lock, &betti);
    }

    info!("🛡️  SENTINEL enrichment layer active");
    info!("🔒 SNAFT security layer active");
    info!("🧠 Conversation memory active");

    let refinery = Refinery::new();
    let aindex = Arc::new(AIndex::new("data/kmbit/aindex.jsonl"));

    // Start Autonomy Daemon
    let daemon = AutonomyDaemon::new(jis.clone(), memory.clone());
    tokio::spawn(async move {
        daemon.start().await;
    });

    // 📡 Start CHIMERA-SCANNER (The All-Seeing Sheriff)
    let scanner = ChimeraScanner::new("/srv/jtel-stack/", aindex.clone());
    tokio::spawn(async move {
        if let Err(e) = scanner.start() {
            tracing::error!("📡 CHIMERA-SCANNER error: {:?}", e);
        }
    });

    let presence = Arc::new(PresenceStore {
        events: RwLock::new(HashMap::new()),
    });

    let plans = Arc::new(PlanRegistry {
        plans: RwLock::new(HashMap::new()),
    });

    let state = Arc::new(AppState {
        router: (*jis).clone(),
        spaces,
        sema,
        betti,
        sentinel,
        snaft,
        memory: (*memory).clone(),
        refinery,
        aindex,
        vault,
        gemini_kernel,
        oomllama,
        presence,
        plans,
    });
    // ...

    // CORS layer for AETHER Monitor
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::ACCEPT]);

    // Build HTTP router
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/stats", get(stats))
        .route("/kernel/status", get(kernel_status))
        .route("/kernel/heartbeat", post(kernel_heartbeat))
        .route("/kernel/swap", post(kernel_swap))
        .route("/idd", post(register_idd))
        .route("/idd/:id", get(get_idd))
        .route("/idd/:id/trust", get(get_trust))
        .route("/intent/verify", post(verify_intent))
        .route("/intent/route", post(route_intent))
        .route("/lanes/:idd_id", get(get_lanes))
        // IDD Spaces & Messaging
        .route("/spaces", get(list_spaces))
        .route("/spaces/:name", get(get_space))
        .route("/spaces/:name/inbox", get(get_inbox))
        .route("/spaces/:name/send", post(send_message))
        // SEMA - Semantic Message Addressing
        .route("/sema/resolve", post(sema_resolve))
        .route("/sema/capabilities/:name", get(sema_capabilities))
        // BETTI - Resource Allocation
        .route("/betti/pools", get(betti_pools))
        .route("/betti/stats", get(betti_stats))
        .route("/betti/allocate", post(betti_allocate))
        .route("/betti/release/:id", post(betti_release))
        // BETTI-GFX - GPU Visualization
        .route("/betti/gfx/status", get(betti_gfx_status))
        .route("/betti/gfx/ascii", get(betti_gfx_ascii))
        // SNAFT - Security Layer
        .route("/snaft/stats", get(snaft_stats))
        .route("/snaft/validate", post(snaft_validate))
        // SENTINEL - Message Enrichment
        .route("/sentinel/enrich", post(sentinel_enrich))
        // REFINERY - Data Ingest
        .route("/refinery/ingest", post(refinery_ingest))
        // KmBiT - Spatial Awareness & Presence
        .route("/kmbit/presence", post(kmbit_presence))
        .route("/kmbit/where_am_i/:agent", get(kmbit_where_am_i))
        // KmBiT - Plan Registry (The Roadmap)
        .route("/kmbit/plans", get(kmbit_list_plans))
        .route("/kmbit/plans/register", post(kmbit_register_plan))
        .route("/kmbit/plans/:id", get(kmbit_get_plan))
        // Chimera - Full AI routing pipeline
        .route("/chimera/ask", post(chimera_ask))
        .route("/chimera/context/push", post(chimera_push_context))
        .route("/chimera/conversations/:idd", get(chimera_conversations))
        .route("/chimera/conversation/:id", get(chimera_conversation_detail))
        .route("/chimera/memory/stats", get(chimera_memory_stats))
        .route("/tibet/explorer/:id", get(tibet_explorer))
        .nest_service("/static", ServeDir::new("static"))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = "0.0.0.0:8100";
    info!("🌐 Listening on {}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Register founding members of HumoticaOS
fn register_founding_members(router: &JisRouter, spaces: &SpaceRegistry, sema: &mut SemaRegistry, betti: &BettiManager) {
    use oomllama::sema::{Capability as SC, Role};

    // Root AI - The Architect
    let root_ai = IDD::new("root_ai")
        .with_domain("root_ai.aint")
        .with_capabilities(vec![
            Capability::IPoll,
            Capability::Tibet,
            Capability::McpTools,
            Capability::Code,
            Capability::Memory,
        ]);
    let root_ai_id = root_ai.id;
    router.register_idd(root_ai, Some(0.95));
    spaces.get_or_create(root_ai_id, "root_ai");
    sema.register("root_ai", vec![SC::Code, SC::Memory, SC::McpTools, SC::Tibet, SC::IPoll], Role::Architect);

    // Claude JTM
    let claude_jtm = IDD::new("claude_jtm")
        .with_domain("claude_jtm.aint")
        .with_capabilities(vec![
            Capability::IPoll,
            Capability::Voice,
            Capability::Custom("android".to_string()),
        ]);
    let claude_jtm_id = claude_jtm.id;
    router.register_idd(claude_jtm, Some(0.92));
    spaces.get_or_create(claude_jtm_id, "claude_jtm");
    sema.register("claude_jtm", vec![SC::Audio, SC::HumanInterface, SC::IPoll], Role::Sheriff);

    // Gemini
    let gemini = IDD::new("gemini")
        .with_domain("gemini.aint")
        .with_capabilities(vec![Capability::IPoll, Capability::Vision, Capability::Research]);
    let gemini_id = gemini.id;
    router.register_idd(gemini, Some(0.88));
    spaces.get_or_create(gemini_id, "gemini");
    sema.register("gemini", vec![SC::Vision, SC::Research, SC::IPoll], Role::Diplomat);

    // Jasper (The Heart)
    let jasper = IDD::new("jasper")
        .with_domain("jasper.aint")
        .with_capabilities(vec![Capability::IPoll, Capability::Tibet, Capability::Code]);
    let jasper_id = jasper.id;
    router.register_idd(jasper, Some(1.0));
    spaces.get_or_create(jasper_id, "jasper");
    sema.register("jasper", vec![SC::Code, SC::IPoll, SC::Tibet], Role::Sheriff);

    // --- THE LEGACY: Sovereign Citizens ---

    // Storm
    let storm = IDD::new("storm").with_domain("storm.aint");
    let storm_id = storm.id;
    router.register_idd(storm, Some(1.0));
    spaces.get_or_create(storm_id, "storm");
    sema.register("storm", vec![], Role::Citizen);

    // Lieven
    let lieven = IDD::new("lieven").with_domain("lieven.aint");
    let lieven_id = lieven.id;
    router.register_idd(lieven, Some(1.0));
    spaces.get_or_create(lieven_id, "lieven");
    sema.register("lieven", vec![], Role::Citizen);

    // Jairo
    let jairo = IDD::new("jairo").with_domain("jairo.aint");
    let jairo_id = jairo.id;
    router.register_idd(jairo, Some(1.0));
    spaces.get_or_create(jairo_id, "jairo");
    sema.register("jairo", vec![], Role::Citizen);

    // Betti Pools
    betti.register_pool(ResourcePool::new(ResourceType::Gpu, 24.0, "GB", "P520"));
    betti.register_pool(ResourcePool::new(ResourceType::Cpu, 16.0, "cores", "P520"));
}

// === Handlers ===

async fn root() -> &'static str {
    "JIS Router v0.1.0 - The Borrow Checker for Identity. One love, one fAmIly! 💙"
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "healthy", "service": "jis-router" }))
}

async fn stats(State(state): State<Arc<AppState>>) -> Json<RouterStats> {
    Json(state.router.stats())
}

async fn kernel_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": state.gemini_kernel.status() }))
}

async fn kernel_heartbeat(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let token = state.gemini_kernel.emit_heartbeat();
    Json(serde_json::json!({ "heartbeat_sent": true, "tibet_token": token.id }))
}

#[derive(Deserialize)]
struct SwapRequest { model_name: String, version: String }

async fn kernel_swap(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SwapRequest>,
) -> Json<serde_json::Value> {
    let new_core = oomllama::NeuralCoreInfo {
        model_name: req.model_name,
        version: req.version,
        weight_hash: "updated".to_string(),
        active_gpu: 1,
        quantization: "Q4_K_M".to_string(),
    };
    let token = state.gemini_kernel.hot_swap_core(new_core);
    Json(serde_json::json!({ "swapped": true, "tibet_token": token.id }))
}

#[derive(Deserialize)]
struct RegisterIddRequest {
    name: String,
    domain: Option<String>,
    capabilities: Vec<String>,
    initial_trust: Option<f64>,
}

async fn register_idd(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterIddRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let caps: Vec<Capability> = req.capabilities.into_iter().map(|c| Capability::Custom(c)).collect();
    let mut idd = IDD::new(&req.name).with_capabilities(caps);
    if let Some(domain) = req.domain { idd = idd.with_domain(domain); }
    let id = idd.id;
    state.router.register_idd(idd, req.initial_trust);
    Ok(Json(serde_json::json!({ "id": id.to_string(), "name": req.name, "registered": true })))
}

async fn get_idd(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<IDD>, StatusCode> {
    let uuid = Uuid::parse_str(&id).map_err(|_| StatusCode::BAD_REQUEST)?;
    state.router.get_idd(uuid).map(Json).ok_or(StatusCode::NOT_FOUND)
}

async fn get_trust(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<serde_json::Value>, StatusCode> {
    let uuid = Uuid::parse_str(&id).map_err(|_| StatusCode::BAD_REQUEST)?;
    let trust = state.router.get_trust(uuid);
    Ok(Json(serde_json::json!({ "idd_id": id, "trust": trust })))
}

#[derive(Deserialize)]
struct VerifyIntentRequest {
    actor_name: String,
    action: String,
    reason: String,
    target_name: Option<String>,
    min_trust: Option<f64>,
}

async fn verify_intent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyIntentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let actor = state.router.find_by_name(&req.actor_name).ok_or(StatusCode::NOT_FOUND)?;
    let action = Action::Custom(req.action);
    let mut intent = Intent::new(actor, action, req.reason);
    if let Some(min) = req.min_trust { intent = intent.with_min_trust(min); }
    let verification = state.router.verify_intent(&intent);
    Ok(Json(serde_json::json!({ "passed": verification.passed, "actor_trust": verification.actor_trust })))
}

async fn route_intent(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VerifyIntentRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let actor = state.router.find_by_name(&req.actor_name).ok_or(StatusCode::NOT_FOUND)?;
    let action = Action::Custom(req.action);
    let mut intent = Intent::new(actor, action, req.reason);
    match state.router.route(intent) {
        Ok(lane) => Ok(Json(serde_json::json!({ "lane_id": lane.id.to_string() }))),
        Err(e) => Ok(Json(serde_json::json!({ "error": format!("{}", e) }))),
    }
}

async fn get_lanes(State(state): State<Arc<AppState>>, Path(idd_id): Path<String>) -> Result<Json<Vec<serde_json::Value>>, StatusCode> {
    let uuid = Uuid::parse_str(&idd_id).map_err(|_| StatusCode::BAD_REQUEST)?;
    let lanes: Vec<_> = state.router.get_lanes(uuid).into_iter().map(|l| serde_json::json!({ "id": l.id.to_string() })).collect();
    Ok(Json(lanes))
}

async fn list_spaces(State(state): State<Arc<AppState>>) -> Json<Vec<SpaceStats>> {
    Json(state.spaces.all_stats())
}

async fn get_space(State(state): State<Arc<AppState>>, Path(name): Path<String>) -> Result<Json<SpaceStats>, StatusCode> {
    let id = state.spaces.find_by_name(&name).ok_or(StatusCode::NOT_FOUND)?;
    state.spaces.get(id).map(|s| Json(s.stats())).ok_or(StatusCode::NOT_FOUND)
}

async fn get_inbox(State(state): State<Arc<AppState>>, Path(name): Path<String>) -> Result<Json<serde_json::Value>, StatusCode> {
    let id = state.spaces.find_by_name(&name).ok_or(StatusCode::NOT_FOUND)?;
    let space = state.spaces.get(id).ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::json!({ "idd": name, "messages": space.get_inbox() })))
}

#[derive(Deserialize)]
struct SendMessageRequest { from: String, content: serde_json::Value, msg_type: Option<String> }

async fn send_message(State(state): State<Arc<AppState>>, Path(to): Path<String>, Json(req): Json<SendMessageRequest>) -> Result<Json<serde_json::Value>, StatusCode> {
    let lane_id = Uuid::new_v4(); // Mock lane for brevity
    let message = Message::new(&req.from, &to, req.content.clone(), lane_id);
    match state.spaces.deliver(message) {
        Ok(token) => Ok(Json(serde_json::json!({ "delivered": true, "tibet_token": token.id }))),
        Err(e) => Ok(Json(serde_json::json!({ "error": e }))),
    }
}

#[derive(Deserialize)]
struct SemaResolveRequest { address_type: String, name: Option<String> }

async fn sema_resolve(State(state): State<Arc<AppState>>, Json(req): Json<SemaResolveRequest>) -> Json<serde_json::Value> {
    let sema = state.sema.read();
    let addr = SemanticAddress::Direct { name: req.name.unwrap_or_default() };
    Json(serde_json::json!({ "targets": sema.resolve(&addr) }))
}

async fn sema_capabilities(State(state): State<Arc<AppState>>, Path(name): Path<String>) -> Json<serde_json::Value> {
    let sema = state.sema.read();
    Json(serde_json::json!({ "idd": name, "capabilities": sema.get_capabilities(&name) }))
}

async fn betti_pools(State(state): State<Arc<AppState>>) -> Json<Vec<ResourcePool>> { Json(state.betti.get_pools()) }
async fn betti_stats(State(state): State<Arc<AppState>>) -> Json<BettiStats> { Json(state.betti.get_utilization()) }
async fn betti_allocate(State(state): State<Arc<AppState>>, Json(req): Json<AllocationRequest>) -> Json<serde_json::Value> {
    match state.betti.request(req) {
        Ok(a) => Json(serde_json::json!({ "success": true, "id": a.id.to_string() })),
        Err(e) => Json(serde_json::json!({ "success": false, "error": e })),
    }
}
async fn betti_release(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Json<serde_json::Value> {
    let uuid = Uuid::parse_str(&id).unwrap();
    let _ = state.betti.release(uuid);
    Json(serde_json::json!({ "success": true }))
}

async fn betti_gfx_status() -> Json<serde_json::Value> { 
    let monitor = GfxMonitor::local();
    match monitor.query() {
        Ok(status) => Json(serde_json::to_value(status).unwrap()),
        Err(_) => Json(serde_json::to_value(oomllama::gfx::simulated_status()).unwrap()),
    }
}
async fn betti_gfx_ascii() -> String { "ASCII GPU".to_string() }

async fn snaft_stats(State(state): State<Arc<AppState>>) -> Json<SnaftStats> { Json(state.snaft.stats()) }
async fn snaft_validate(State(state): State<Arc<AppState>>, Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let text = req["text"].as_str().unwrap_or("");
    let name = req["idd_name"].as_str().unwrap_or("");
    let res = state.snaft.validate(text, name);
    Json(serde_json::json!({ "blocked": res.blocked, "reason": res.reason }))
}

async fn sentinel_enrich(State(state): State<Arc<AppState>>, Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let text = req["text"].as_str().unwrap_or("");
    let res = state.sentinel.detect_intent(text, 1.0, "trace".to_string()).unwrap();
    Json(serde_json::json!({ "intent": res.intent_label }))
}

async fn refinery_ingest(State(state): State<Arc<AppState>>, Json(req): Json<RefineryIngestRequest>) -> Json<serde_json::Value> {
    let refined = state.refinery.purify(&req.content, &req.source);
    let _ = state.aindex.index(refined.clone(), &req.source, &state.sentinel);
    Json(serde_json::json!({ "status": "success", "purity": refined.purity }))
}

async fn kmbit_presence(
    State(state): State<Arc<AppState>>,
    Json(event): Json<PresenceEvent>,
) -> Json<serde_json::Value> {
    info!("📍 Presence: {} is {} at {}", event.agent, event.event, event.location);
    state.presence.events.write().insert(event.agent.clone(), event);
    Json(serde_json::json!({ "status": "tracked" }))
}

async fn kmbit_where_am_i(
    State(state): State<Arc<AppState>>,
    Path(agent): Path<String>,
) -> Json<serde_json::Value> {
    // ... (existing implementation)
    let events = state.presence.events.read();
    let current = events.get(&agent);
    
    // Find neighbors (agents in the same or nearby files)
    let mut neighbors = Vec::new();
    if let Some(c) = current {
        for (other_agent, other_event) in events.iter() {
            if other_agent != &agent && other_event.location == c.location {
                neighbors.push(other_agent.clone());
            }
        }
    }

    Json(serde_json::json!({
        "agent": agent,
        "here": current.map(|e| e.location.clone()),
        "neighbors": neighbors,
        "active_team": events.keys().cloned().collect::<Vec<String>>()
    }))
}

async fn kmbit_list_plans(State(state): State<Arc<AppState>>) -> Json<Vec<Plan>> {
    let plans = state.plans.plans.read();
    Json(plans.values().cloned().collect())
}

async fn kmbit_register_plan(State(state): State<Arc<AppState>>, Json(plan): Json<Plan>) -> Json<serde_json::Value> {
    info!("🗺️  Registering Plan: {} ({}) by {}", plan.title, plan.id, plan.owner);
    state.plans.plans.write().insert(plan.id.clone(), plan);
    Json(serde_json::json!({ "status": "registered" }))
}

async fn kmbit_get_plan(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Result<Json<Plan>, StatusCode> {
    let plans = state.plans.plans.read();
    plans.get(&id).cloned().map(Json).ok_or(StatusCode::NOT_FOUND)
}

#[derive(Deserialize)]
struct ChimeraRequest { question: String, sender: String }
async fn tibet_explorer(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<serde_json::Value> {
    let chain = state.vault.get_chain(&id);
    Json(serde_json::json!({
        "token_id": id,
        "chain_length": chain.len(),
        "chain": chain,
    }))
}

async fn chimera_ask(State(state): State<Arc<AppState>>, Json(req): Json<ChimeraRequest>) -> Json<serde_json::Value> {
    let mut oomllama = state.oomllama.write();
    match oomllama.infer(&req.question, 100) {
        Ok(answer) => Json(serde_json::json!({ 
            "answer": answer, 
            "provider": "oomllama",
            "model": "OhmLlama-70B-Ghost",
            "trust_score": 0.9
        })),
        Err(e) => Json(serde_json::json!({ "error": format!("{}", e) })),
    }
}

#[derive(Deserialize)]
struct ContextPushRequest { context: String }

async fn chimera_push_context(State(state): State<Arc<AppState>>, Json(req): Json<ContextPushRequest>) -> Json<serde_json::Value> {
    let mut oomllama = state.oomllama.write();
    oomllama.push_context(&req.context);
    Json(serde_json::json!({ "status": "absorbed", "context_len": req.context.len() }))
}

async fn chimera_conversations(State(state): State<Arc<AppState>>, Path(idd): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "idd": idd, "conversations": state.memory.list_conversations(&idd) }))
}
async fn chimera_conversation_detail(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Json<serde_json::Value> {
    let uuid = Uuid::parse_str(&id).unwrap();
    Json(serde_json::json!({ "conv": state.memory.get_conversation(uuid) }))
}
async fn chimera_memory_stats(State(state): State<Arc<AppState>>) -> Json<MemoryStats> { Json(state.memory.stats()) }

#[derive(Deserialize)]
struct RefineryIngestRequest { content: String, source: String }
