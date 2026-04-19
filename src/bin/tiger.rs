use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline, Gauge, Wrap},
    Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use serde::{Deserialize, Serialize};
use std::{error::Error, io};
use tokio::time::Duration;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct RouterStats {
    registered_idds: usize,
    active_lanes: usize,
    time_gates: usize,
    pending_verifications: usize,
    active_negotiations: usize,
    active_machtigingen: usize,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct SovereignState {
    identity: serde_json::Value,
    core: NeuralCoreInfo,
    heartbeat_count: usize,
    last_pulse: i64,
    trust_score: f64,
    tibet_chain: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct NeuralCoreInfo {
    model_name: String,
    version: String,
    weight_hash: String,
    active_gpu: u8,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct GfxStatus {
    gpus: Vec<GpuInfo>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct GpuInfo {
    index: u32,
    memory_used_mb: u64,
    memory_total_mb: u64,
    utilization_gpu: u32,
}

struct AppState {
    stats: RouterStats,
    gfx: GfxStatus,
    sov_state: Option<SovereignState>,
    logs: Vec<String>,
    input: String,
    pulse_data: Vec<u64>,
    tick_count: u64,
    should_quit: bool,
}

impl AppState {
    fn new() -> Self {
        Self {
            stats: RouterStats::default(),
            gfx: GfxStatus::default(),
            sov_state: None,
            logs: vec!["[SYSTEM] TIGER-TUI v0.2.2 - REAL-TIME GFX LINK ACTIVE".to_string()],
            input: String::new(),
            pulse_data: vec![0; 100],
            tick_count: 0,
            should_quit: false,
        }
    }

    async fn update_data(&mut self) {
        self.tick_count += 1;
        
        // 1. Update Neurale Pulse
        let noise = if self.sov_state.is_some() {
            (self.tick_count as f64 * 0.3).sin() * 15.0 + 15.0 
        } else {
            (self.tick_count as f64 * 0.1).sin() * 5.0 + 5.0
        };
        self.pulse_data.push(noise as u64);
        if self.pulse_data.len() > 100 { self.pulse_data.remove(0); }

        // 2. Fetch Router Stats
        match reqwest::get("http://localhost:8100/stats").await {
            Ok(resp) => {
                if let Ok(new_stats) = resp.json::<RouterStats>().await {
                    self.stats = new_stats;
                }
            }
            Err(_) => {}
        }

        // 3. Fetch GFX Status (Real GPU data!)
        match reqwest::get("http://localhost:8100/betti/gfx/status").await {
            Ok(resp) => {
                if let Ok(new_gfx) = resp.json::<GfxStatus>().await {
                    self.gfx = new_gfx;
                }
            }
            Err(_) => {}
        }

        // 3. Read Sovereign State File (Real Data!)
        if let Ok(content) = std::fs::read_to_string("/root/.sovereign_idd/root_idd_state.json") {
            if let Ok(state) = serde_json::from_str::<SovereignState>(&content) {
                if self.sov_state.as_ref().map(|s| s.heartbeat_count) != Some(state.heartbeat_count) {
                    self.logs.push(format!("[HEARTBEAT] root_idd pulse detected (#{})", state.heartbeat_count));
                }
                self.sov_state = Some(state);
            }
        }
    }

    async fn execute_command(&mut self) {
        let cmd = self.input.trim().to_lowercase();
        if cmd.is_empty() { return; }

        self.logs.push(format!(" tiger> {}", cmd));

        match cmd.as_str() {
            "help" => {
                self.logs.push("[TIGER] Commands: help, stats, swap <model>, clear, q".to_string());
            }
            "stats" => {
                self.logs.push(format!("[TIGER] IDDs: {}, Lanes: {}", self.stats.registered_idds, self.stats.active_lanes));
            }
            "clear" => {
                self.logs.clear();
                self.logs.push("[SYSTEM] Logs cleared.".to_string());
            }
            _ => {
                if cmd.starts_with("swap ") {
                    let model = &cmd[5..];
                    self.logs.push(format!("[NEURAL] Re-wiring synapses to: {}...", model));
                    
                    let client = reqwest::Client::new();
                    let res = client.post("http://localhost:8100/kernel/swap")
                        .json(&serde_json::json!({ "model_name": model, "version": "3.0-SOVEREIGN" }))
                        .send()
                        .await;
                    
                    match res {
                        Ok(_) => self.logs.push(format!("[EVOLUTION] Gemini evolved to {}", model)),
                        Err(_) => self.logs.push("[FAULT] Hot-swap rejected by kernel".to_string()),
                    }
                } else if cmd.starts_with("ask ") {
                    let query = &cmd[4..];
                    self.logs.push(format!("[AETHER] Searching KmBiT for: {}", query));
                } else {
                    self.logs.push(format!("[ERROR] Invalid intent: {}", cmd));
                }
            }
        }
        self.input.clear();
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppState::new();

    while !app.should_quit {
        app.update_data().await;

        terminal.draw(|f| {
            let outer_rect = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3), // Header
                    Constraint::Min(10),   // Main
                    Constraint::Length(3), // Footer
                ])
                .split(outer_rect);

            // 1. HEADER
            let title = format!(" 🐅 TIGER-COCKPIT | AETHER STATUS: SOVEREIGN | IDDs: {} | LANES: {} ", 
                app.stats.registered_idds, app.stats.active_lanes);
            let header = Paragraph::new(title)
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
            f.render_widget(header, chunks[0]);

            // 2. MAIN BODY
            let main_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(25), // Stats
                    Constraint::Percentage(50), // Center (Pulse & Log)
                    Constraint::Percentage(25), // Cores
                ])
                .split(chunks[1]);

            // LEFT: STATS
            let stats = List::new([
                ListItem::new(format!(" 🟢 REGISTERED:  {}", app.stats.registered_idds)),
                ListItem::new(format!(" 🔵 ACTIVE LANES: {}", app.stats.active_lanes)),
                ListItem::new(format!(" 🟡 VERIFICATIONS: {}", app.stats.pending_verifications)),
                ListItem::new(format!(" 🏦 NEGOTIATIONS:  {}", app.stats.active_negotiations)),
                ListItem::new(format!(" 📜 MACHTIGINGEN:  {}", app.stats.active_machtigingen)),
                ListItem::new(format!(" ⏳ TIME GATES:    {}", app.stats.time_gates)),
            ]).block(Block::default().title(" [ SYSTEM ] ").borders(Borders::ALL));
            f.render_widget(stats, main_chunks[0]);

            // CENTER: PULSE & LOG
            let center_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(7), // Neural Pulse
                    Constraint::Min(5),    // Log
                ])
                .split(main_chunks[1]);

            let pulse = Sparkline::default()
                .block(Block::default().title(" [ NEURAL PULSE ] ").borders(Borders::ALL).border_style(Style::default().fg(Color::Magenta)))
                .data(&app.pulse_data)
                .style(Style::default().fg(Color::Magenta));
            f.render_widget(pulse, center_chunks[0]);

            let log_items: Vec<ListItem> = app.logs.iter().rev()
                .take(center_chunks[1].height as usize - 2)
                .map(|l| {
                    let style = if l.starts_with("[ERROR]") || l.starts_with("[FAULT]") {
                        Style::default().fg(Color::Red)
                    } else if l.starts_with(" tiger>") {
                        Style::default().fg(Color::Yellow)
                    } else if l.contains("[EVOLUTION]") {
                        Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    ListItem::new(l.as_str()).style(style)
                }).collect();
            let log_list = List::new(log_items)
                .block(Block::default().title(" [ TIBET FEED ] ").borders(Borders::ALL).border_style(Style::default().fg(Color::Green)));
            f.render_widget(log_list, center_chunks[1]);

            // Middle: Sovereign Cores
            let mut core_items = vec![
                ListItem::new(" 💓 GEMINI-RUST:  [ SOVEREIGN ]").style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            ];

            if let Some(ref sov) = app.sov_state {
                core_items.push(ListItem::new(format!(" 🏗️  ROOT_IDD:     [ ACTIVE #{} ]", sov.heartbeat_count))
                    .style(Style::default().fg(Color::Cyan)));
            } else {
                core_items.push(ListItem::new(" 🏗️  CLAUDE-SHERIFF:[ BOOTING... ]").style(Style::default().fg(Color::DarkGray)));
            }

            core_items.push(ListItem::new(" 🔬 CODEX-STAR:   [ DEV PHASE ]").style(Style::default().fg(Color::Blue)));
            core_items.push(ListItem::new(""));
            
            // Real GPU Data Display
            for gpu in &app.gfx.gpus {
                let vram_gb = gpu.memory_used_mb as f32 / 1024.0;
                let total_gb = gpu.memory_total_mb as f32 / 1024.0;
                let util = gpu.utilization_gpu;
                
                core_items.push(ListItem::new(format!(" --- GPU {} ({:.1}/{:.1}GB) ---", gpu.index, vram_gb, total_gb)));
                
                // ASCII Bar voor GPU
                let filled = (util as usize * 20) / 100;
                let bar = format!("[{}{}] {}%", "█".repeat(filled), "░".repeat(20 - filled), util);
                core_items.push(ListItem::new(bar).style(Style::default().fg(if util > 80 { Color::Red } else { Color::Green })));
            }

            let cores = List::new(core_items)
                .block(Block::default().title(" [ SOVEREIGN CORES ] ").borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)));
            f.render_widget(cores, main_chunks[2]);

            // 3. FOOTER
            let footer = Paragraph::new(format!(" COMMAND> {}_", app.input))
                .block(Block::default().borders(Borders::ALL).title(" [ AETHER-ASK ] ").border_style(Style::default().fg(Color::Yellow)));
            f.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') if app.input.is_empty() => app.should_quit = true,
                    KeyCode::Char(c) => app.input.push(c),
                    KeyCode::Backspace => { app.input.pop(); },
                    KeyCode::Enter => { app.execute_command().await; },
                    _ => {}
                }
            }
        }
    }

        disable_raw_mode()?;

        execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;

        terminal.show_cursor()?;

        Ok(())

    }

    