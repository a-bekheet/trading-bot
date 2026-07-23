import {
  Activity,
  Bot,
  Boxes,
  BrainCircuit,
  ChevronRight,
  CircleDollarSign,
  Database,
  FlaskConical,
  Gauge,
  GraduationCap,
  History,
  LayoutDashboard,
  LoaderCircle,
  Play,
  Power,
  RefreshCw,
  RotateCcw,
  Search,
  ServerCog,
  ShieldCheck,
  Square,
  WalletCards,
  Zap,
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useState } from "react";
import { api, compact, money, percent, titleCase } from "./api";

type Page =
  | "command"
  | "agents"
  | "models"
  | "training"
  | "trade"
  | "portfolio"
  | "research"
  | "system";

type Json = Record<string, any>;

const navigation: { id: Page; label: string; icon: typeof Activity }[] = [
  { id: "command", label: "Command Center", icon: LayoutDashboard },
  { id: "agents", label: "Agents", icon: Bot },
  { id: "models", label: "Models", icon: BrainCircuit },
  { id: "training", label: "Training", icon: GraduationCap },
  { id: "trade", label: "Trade", icon: CircleDollarSign },
  { id: "portfolio", label: "Portfolio", icon: WalletCards },
  { id: "research", label: "Research", icon: FlaskConical },
  { id: "system", label: "System", icon: ServerCog },
];

function useResource<T>(path: string, refreshKey = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let current = true;
    setLoading(true);
    api<T>(path)
      .then((payload) => current && setData(payload))
      .catch((problem: Error) => current && setError(problem.message))
      .finally(() => current && setLoading(false));
    return () => {
      current = false;
    };
  }, [path, refreshKey]);
  return { data, error, loading };
}

function StatusDot({ state }: { state: string }) {
  const normalized = state.toLowerCase();
  const tone =
    ["active", "running", "complete", "up_to_date", "regular", "ok"].some(
      (value) => normalized.includes(value),
    )
      ? "good"
      : ["error", "failed", "degraded"].some((value) =>
            normalized.includes(value),
          )
        ? "bad"
        : "warn";
  return <span className={`status-dot ${tone}`} />;
}

function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

function Metric({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  detail?: string;
  icon?: typeof Activity;
}) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
        {Icon && <Icon size={17} />}
      </div>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function LoadingState() {
  return (
    <div className="state-box">
      <LoaderCircle className="spin" size={24} />
      <span>Loading saved application state…</span>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return <div className="state-box error">{message}</div>;
}

function EmptyState({ children }: { children: ReactNode }) {
  return <div className="state-box">{children}</div>;
}

function JobStrip({ jobs }: { jobs: Json[] }) {
  if (!jobs.length) return <EmptyState>No control jobs have run in this session.</EmptyState>;
  return (
    <div className="job-list">
      {jobs.slice(0, 6).map((job) => (
        <div className="job-row" key={job.id}>
          <StatusDot state={job.status} />
          <div>
            <strong>{job.label}</strong>
            <small>{job.id} · {titleCase(job.status)}</small>
          </div>
          <code>{job.return_code ?? "—"}</code>
        </div>
      ))}
    </div>
  );
}

function CommandCenter({
  refreshKey,
  refresh,
}: {
  refreshKey: number;
  refresh: () => void;
}) {
  const { data, loading, error } = useResource<Json>("/api/overview", refreshKey);
  const [busy, setBusy] = useState("");
  const runAction = async (service: string, action: string) => {
    if (
      ["stop", "restart"].includes(action) &&
      !window.confirm(`${titleCase(action)} ${service.replace("_", " ")}?`)
    ) return;
    setBusy(`${service}:${action}`);
    try {
      await api(`/api/services/${service}/actions`, {
        method: "POST",
        body: JSON.stringify({ action }),
      });
      window.setTimeout(refresh, 500);
    } catch (problem) {
      window.alert((problem as Error).message);
    } finally {
      setBusy("");
    }
  };
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  const market = data.market ?? {};
  const services = data.services ?? [];
  return (
    <>
      <PageHeader
        eyebrow="OPERATIONS"
        title="Command Center"
        description="Run the data, training, and paper-agent stack from one auditable control surface."
        actions={
          <button className="button secondary" onClick={refresh}>
            <RefreshCw size={16} /> Refresh state
          </button>
        }
      />
      <section className="metric-grid four">
        <Metric
          label="Service health"
          value={`${data.service_summary.healthy}/${data.service_summary.total}`}
          detail="local background services"
          icon={ServerCog}
        />
        <Metric
          label="Market"
          value={market.session?.provider_state ?? "Unknown"}
          detail={market.symbol ? `${market.symbol} · ${money(market.underlying_price)}` : "No snapshots"}
          icon={Activity}
        />
        <Metric
          label="Paper agents"
          value={`${data.agents.active}/${data.agents.total}`}
          detail={`${data.agents.decisions} decisions · ${data.agents.executions} fills`}
          icon={Bot}
        />
        <Metric
          label="Manual paper cash"
          value={money(data.account.cash)}
          detail="isolated from agent accounts"
          icon={WalletCards}
        />
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Runtime controls</h2>
            <p>Every action creates a serialized job with captured output.</p>
          </div>
          <span className="safety-label"><ShieldCheck size={15} /> Paper only</span>
        </div>
        <div className="service-grid">
          {services.map((service: Json) => (
            <article className="service-card" key={service.id}>
              <div className="service-title">
                <div className="service-icon">
                  {service.id === "collector" ? <Database /> : service.id === "training" ? <GraduationCap /> : <Bot />}
                </div>
                <div>
                  <h3>{service.label}</h3>
                  <p>{service.description}</p>
                </div>
              </div>
              <div className="service-state">
                <span><StatusDot state={service.status} />{titleCase(service.status)}</span>
                <small>{service.last_heartbeat_at ?? "No heartbeat recorded"}</small>
              </div>
              {service.message && <div className="service-message">{service.message}</div>}
              <div className="control-row">
                <button
                  className="icon-button positive"
                  title="Start service"
                  disabled={!!busy}
                  onClick={() => runAction(service.id, "start")}
                ><Play size={16} /> Start</button>
                <button
                  className="icon-button"
                  title="Run one cycle"
                  disabled={!!busy}
                  onClick={() => runAction(service.id, "run_once")}
                ><Zap size={16} /> Once</button>
                <button
                  className="icon-button"
                  title="Restart service"
                  disabled={!!busy}
                  onClick={() => runAction(service.id, "restart")}
                ><RotateCcw size={16} /></button>
                <button
                  className="icon-button danger"
                  title="Stop service"
                  disabled={!!busy}
                  onClick={() => runAction(service.id, "stop")}
                ><Square size={15} /></button>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="split-layout">
        <div className="panel">
          <div className="panel-heading"><h2>Recent control jobs</h2><History size={18} /></div>
          <JobStrip jobs={data.jobs ?? []} />
        </div>
        <div className="panel attention-panel">
          <div className="panel-heading"><h2>Operator attention</h2><Gauge size={18} /></div>
          <ul className="attention-list">
            {services.filter((item: Json) => !item.healthy).length ? (
              services.filter((item: Json) => !item.healthy).map((item: Json) => (
                <li key={item.id}><StatusDot state={item.status} /><span><strong>{item.label}</strong>{item.message || ` is ${titleCase(item.status)}`}</span></li>
              ))
            ) : (
              <li><StatusDot state="ok" /><span><strong>System nominal</strong>All service heartbeats report an expected state.</span></li>
            )}
          </ul>
        </div>
      </section>
    </>
  );
}

function TrainingPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/training", refreshKey);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [episodes, setEpisodes] = useState(3);
  const [hiddenSize, setHiddenSize] = useState(16);
  const [sequenceLength, setSequenceLength] = useState(4);
  const [maxSteps, setMaxSteps] = useState(16);
  const [launching, setLaunching] = useState(false);
  const [job, setJob] = useState<Json | null>(null);
  useEffect(() => {
    if (data?.defaults?.symbols && !symbols.length) setSymbols(data.defaults.symbols);
  }, [data, symbols.length]);
  const toggle = (symbol: string) =>
    setSymbols((current) =>
      current.includes(symbol)
        ? current.filter((value) => value !== symbol)
        : [...current, symbol],
    );
  const launch = async (event: FormEvent) => {
    event.preventDefault();
    if (!window.confirm(`Launch 12 candidates × 3 seeds for ${symbols.join(", ")}?`)) return;
    setLaunching(true);
    try {
      setJob(
        await api("/api/training/runs", {
          method: "POST",
          body: JSON.stringify({
            symbols,
            episodes,
            hidden_size: hiddenSize,
            sequence_length: sequenceLength,
            max_steps: maxSteps,
          }),
        }),
      );
    } catch (problem) {
      window.alert((problem as Error).message);
    } finally {
      setLaunching(false);
    }
  };
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  const readiness = data.readiness ?? [];
  const ready = readiness.filter((item: Json) => item.Ready === "Yes").length;
  return (
    <>
      <PageHeader
        eyebrow="AGENT DEVELOPMENT"
        title="Training"
        description="Configure, preflight, and launch recurrent and surface-GNN tournaments without leaving the application."
      />
      <section className="metric-grid four">
        <Metric label="Readiness" value={`${ready}/${readiness.length}`} detail="tickers passing strict tail checks" icon={ShieldCheck} />
        <Metric label="Candidates" value={data.defaults.candidate_count_per_ticker} detail="per ticker · validation selected" icon={Boxes} />
        <Metric label="Seeds" value={data.defaults.training_seed_count} detail="every locked candidate" icon={BrainCircuit} />
        <Metric label="Watcher" value={titleCase(data.watcher?.status ?? "not started")} detail={data.watcher?.message ?? "no watcher heartbeat"} icon={Activity} />
      </section>
      <section className="training-layout">
        <form className="panel training-form" onSubmit={launch}>
          <div className="panel-heading">
            <div><h2>New arena run</h2><p>Locked deployable policy surface</p></div>
            <span className="safety-label"><FlaskConical size={14} /> Research demo</span>
          </div>
          <label className="field-label">Tickers</label>
          <div className="ticker-select">
            {data.defaults.symbols.map((symbol: string) => (
              <button
                type="button"
                className={symbols.includes(symbol) ? "ticker active" : "ticker"}
                onClick={() => toggle(symbol)}
                key={symbol}
              >{symbol}</button>
            ))}
          </div>
          <div className="form-grid">
            <label>Episodes<input type="number" min="1" max="100" value={episodes} onChange={(e) => setEpisodes(Number(e.target.value))} /></label>
            <label>Hidden width<select value={hiddenSize} onChange={(e) => setHiddenSize(Number(e.target.value))}>{[8,16,32,64,128].map((value) => <option key={value}>{value}</option>)}</select></label>
            <label>Sequence length<input type="number" min="1" max="64" value={sequenceLength} onChange={(e) => setSequenceLength(Number(e.target.value))} /></label>
            <label>Maximum steps<input type="number" min="2" max="512" value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} /></label>
          </div>
          <div className="run-summary">
            <div><span>Training replicas</span><strong>{symbols.length * 12 * 3}</strong></div>
            <div><span>Model families</span><strong>GRU · LSTM · Mixture · GNN</strong></div>
            <div><span>Selection</span><strong>Validation only</strong></div>
            <div><span>Execution</span><strong>Activation-gated paper</strong></div>
          </div>
          <button className="button primary wide" disabled={launching || !symbols.length}>
            {launching ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
            Review and launch training
          </button>
          {job && <div className="launch-confirm"><StatusDot state={job.status} /><span><strong>Run queued</strong>{job.id} · {job.label}</span></div>}
        </form>
        <div className="panel">
          <div className="panel-heading"><div><h2>Data preflight</h2><p>Strict regular, fresh, executable partitions</p></div><ShieldCheck size={18} /></div>
          <div className="readiness-list">
            {readiness.map((item: Json) => (
              <div className="readiness-row" key={item.Ticker}>
                <div><StatusDot state={item.Ready === "Yes" ? "ok" : "waiting"} /><strong>{item.Ticker}</strong></div>
                <div className="readiness-bar"><span style={{ width: `${Math.min(100, 100 * item["Eligible snapshots"] / Math.max(1, item["Required eligible"]))}%` }} /></div>
                <span>{item["Eligible snapshots"]}/{item["Required eligible"]}</span>
              </div>
            ))}
          </div>
          <div className="callout">
            <ShieldCheck size={18} />
            <p><strong>Held-out boundary remains locked.</strong> Training cannot use test results for model selection or sandbox activation.</p>
          </div>
        </div>
      </section>
      <section className="section panel">
        <div className="panel-heading"><h2>Training jobs</h2><History size={18} /></div>
        <JobStrip jobs={[...(job ? [job] : []), ...(data.jobs ?? [])]} />
      </section>
    </>
  );
}

function AgentsPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/agents", refreshKey);
  const [selected, setSelected] = useState("");
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  const roster = data.roster ?? [];
  const symbol = selected || roster[0]?.Ticker;
  const agent = roster.find((item: Json) => item.Ticker === symbol);
  const deployment = (data.deployments ?? []).find((item: Json) => item.Ticker === symbol);
  const decisions = (data.decisions ?? []).filter((item: Json) => item.Ticker === symbol);
  return (
    <>
      <PageHeader eyebrow="PAPER FLEET" title="Agents" description="Inspect every selected policy, its validation gate, causal decisions, and isolated paper account." />
      {!roster.length ? <EmptyState>No selected agent artifacts are available.</EmptyState> : (
        <section className="master-detail">
          <aside className="master-list">
            <div className="list-search"><Search size={16} /><span>Selected policy fleet</span></div>
            {roster.map((item: Json) => (
              <button className={item.Ticker === symbol ? "master-item active" : "master-item"} onClick={() => setSelected(item.Ticker)} key={item.Ticker}>
                <span className="ticker-mark">{item.Ticker.slice(0,2)}</span>
                <span><strong>{item.Ticker}</strong><small>{item["Research policy"]}</small></span>
                <StatusDot state={item.State} />
              </button>
            ))}
          </aside>
          <div className="detail">
            <div className="detail-hero">
              <div><div className="eyebrow">SELECTED CHECKPOINT</div><h2>{agent.Ticker} · {agent["Research policy"]}</h2><p>{agent.Architecture} · {agent.Algorithm} · {agent["Action policy"]}</p></div>
              <span className={`state-badge ${agent.State.includes("active") ? "good" : "warn"}`}><StatusDot state={agent.State} />{agent.State}</span>
            </div>
            <div className="metric-grid four">
              <Metric label="Held-out return" value={percent(agent["Held-out return"])} detail="fixed winner · research path" />
              <Metric label="Online paper return" value={deployment ? percent(deployment["Paper return"]) : "Pending"} detail={`${deployment?.["Finalized outcomes"] ?? 0} finalized outcomes`} />
              <Metric label="Actor latency" value={`${Number(agent["Median latency (us)"]).toFixed(1)} µs`} detail={agent.Architecture} />
              <Metric label="Validation edge" value={`${Number(agent["Validation edge vs no-op (bp)"]).toFixed(2)} bp`} detail={agent.State.includes("active") ? "activation passed" : "sandbox forced to HOLD"} />
            </div>
            <div className="panel">
              <div className="panel-heading"><div><h2>Decision timeline</h2><p>Research proposal versus sandbox action</p></div><Activity size={18} /></div>
              {!decisions.length ? <EmptyState>No eligible online decisions recorded.</EmptyState> : (
                <div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Research</th><th>Sandbox</th><th>Confidence</th><th>Outcome</th><th>NAV</th></tr></thead><tbody>
                  {decisions.slice(0,20).map((item: Json, index: number) => <tr key={index}><td>{String(item.Timestamp).replace("T"," ").slice(0,19)}</td><td>{item["Research action"]}</td><td>{item["Sandbox action"]}</td><td>{percent(item["Action confidence"],1)}</td><td>{item["Outcome status"]}</td><td>{money(item.NAV)}</td></tr>)}
                </tbody></table></div>
              )}
            </div>
          </div>
        </section>
      )}
    </>
  );
}

function ModelsPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/models", refreshKey);
  const [selected, setSelected] = useState("");
  const [node, setNode] = useState(0);
  const [mode, setMode] = useState<"runtime" | "training">("runtime");
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  const models = data.models ?? [];
  const current = models.find((item: Json) => item.symbol === selected) ?? models[0];
  const spec = current?.structure;
  const stages = spec?.stages ?? [];
  const selectedNode = stages[node] ?? stages[0];
  return (
    <>
      <PageHeader eyebrow="MODEL LABORATORY" title="Models" description="Understand the exact selected architecture, tensor contract, runtime path, and matched challengers." />
      {!current ? <EmptyState>No resolved model contracts found.</EmptyState> : (
        <section className="model-workbench">
          <aside className="model-fleet">
            <div className="list-search"><Boxes size={16} /><span>Selected models</span></div>
            {models.map((item: Json) => (
              <button className={item.symbol === current.symbol ? "model-list-item active" : "model-list-item"} onClick={() => {setSelected(item.symbol); setNode(0);}} key={item.symbol}>
                <div><strong>{item.symbol}</strong><span>{item.structure.temporal_core}</span></div>
                <small>{compact(item.structure.parameters)} params</small>
              </button>
            ))}
          </aside>
          <main className="architecture-canvas">
            <div className="architecture-toolbar">
              <div><div className="eyebrow">SELECTED · FOLD {spec.fold}</div><h2>{current.symbol} · {spec.architecture}</h2></div>
              <div className="segmented"><button className={mode === "runtime" ? "active" : ""} onClick={() => setMode("runtime")}>Runtime</button><button className={mode === "training" ? "active" : ""} onClick={() => setMode("training")}>Training</button></div>
            </div>
            <div className="architecture-flow">
              {stages.map((stage: Json, index: number) => (
                <div className="flow-segment" key={stage.name}>
                  <button className={index === node ? `architecture-node ${stage.kind} active` : `architecture-node ${stage.kind}`} onClick={() => setNode(index)}>
                    <span>{index + 1}</span><strong>{stage.name}</strong><small>{stage.detail}</small>
                  </button>
                  {index < stages.length - 1 && <div className="connector"><ChevronRight size={20} /><code>{index === 0 ? spec.input_width : index === 2 ? spec.actor_output_width : spec.hidden_size || "set"}</code></div>}
                </div>
              ))}
            </div>
            {mode === "training" && (
              <div className="training-heads">
                <div><BrainCircuit size={18} /><span><strong>Critic head</strong>Scalar state value for GAE and value loss</span></div>
                <div><FlaskConical size={18} /><span><strong>{spec.auxiliary_target_count} auxiliary targets</strong>Training-only predictive regularization</span></div>
              </div>
            )}
            <div className="model-stat-strip">
              <div><span>Parameters</span><strong>{spec.parameters.toLocaleString()}</strong></div>
              <div><span>Inputs</span><strong>{spec.active_input_width}</strong></div>
              <div><span>Hidden</span><strong>{spec.hidden_size}</strong></div>
              <div><span>Median actor</span><strong>{spec.median_latency_us.toFixed(1)} µs</strong></div>
              <div><span>Training seeds</span><strong>{spec.training_seed_count}</strong></div>
            </div>
            <div className="panel candidate-panel">
              <div className="panel-heading"><div><h2>Matched candidate fleet</h2><p>Validation scores only · held-out path excluded</p></div><Boxes size={18} /></div>
              <div className="table-wrap"><table><thead><tr><th>State</th><th>Core</th><th>Encoder</th><th>Algorithm</th><th>Decoder</th><th>Parameters</th><th>Validation</th><th>Latency</th></tr></thead><tbody>
                {current.candidates.map((item: Json) => <tr key={item.Model} className={item.Selected === "Winner" ? "winner" : ""}><td>{item.Selected}</td><td>{item.Core}</td><td>{item.Encoder}</td><td>{item.Algorithm}</td><td>{item.Decoder}</td><td>{Number(item.Parameters).toLocaleString()}</td><td>{Number(item["Validation score"]).toFixed(6)}</td><td>{Number(item["Actor latency (us)"]).toFixed(1)} µs</td></tr>)}
              </tbody></table></div>
            </div>
          </main>
          <aside className="inspector">
            <div className="inspector-title"><span>{node + 1}</span><div><div className="eyebrow">COMPONENT</div><h3>{selectedNode?.name}</h3></div></div>
            <p>{selectedNode?.detail}</p>
            <dl>
              <div><dt>Topology</dt><dd>{spec.topology}</dd></div>
              <div><dt>Feature schema</dt><dd>{spec.feature_schema}</dd></div>
              <div><dt>Sequence</dt><dd>{spec.sequence_length} snapshots</dd></div>
              <div><dt>Actor outputs</dt><dd>{spec.actor_output_width} logits</dd></div>
              <div><dt>p95 latency</dt><dd>{spec.p95_latency_us.toFixed(1)} µs</dd></div>
              <div><dt>Checkpoint</dt><dd className="break">{spec.checkpoint_name}</dd></div>
            </dl>
            <div className="callout compact-callout"><ShieldCheck size={17} /><p>{spec.runtime_contract}</p></div>
          </aside>
        </section>
      )}
    </>
  );
}

function TradePage({ refreshKey }: { refreshKey: number }) {
  const [symbol, setSymbol] = useState("");
  const path = `/api/market${symbol ? `?symbol=${symbol}` : ""}`;
  const { data, loading, error } = useResource<Json>(path, refreshKey);
  const [type, setType] = useState("call");
  const [expiration, setExpiration] = useState("");
  const [selected, setSelected] = useState("");
  const [quantity, setQuantity] = useState(1);
  const contracts = (data?.contracts ?? []).filter((item: Json) => item.optionType === type);
  const expirations = [...new Set(contracts.map((item: Json) => item.expiration))] as string[];
  const effectiveExpiration = expiration && expirations.includes(expiration) ? expiration : expirations[0];
  const filtered = contracts.filter((item: Json) => item.expiration === effectiveExpiration);
  const contract = filtered.find((item: Json) => item.contractSymbol === selected) ?? filtered[0];
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  const send = async (side: string) => {
    if (!contract || !window.confirm(`${side.toUpperCase()} ${quantity} ${contract.contractSymbol} in the paper account?`)) return;
    try {
      const fill = await api<Json>("/api/orders", {method:"POST", body:JSON.stringify({side, symbol:data.symbol, contract_symbol:contract.contractSymbol, quantity})});
      window.alert(`Paper ${side} complete · ${fill.quantity} @ ${money(fill.price)}`);
    } catch (problem) { window.alert((problem as Error).message); }
  };
  return (
    <>
      <PageHeader eyebrow="MANUAL PAPER EXECUTION" title="Trade" description="Search the saved executable chain, inspect economics, and submit an explicit simulated fill." />
      <section className="trade-layout">
        <div className="chain-panel panel">
          <div className="chain-toolbar">
            <select value={data.symbol} onChange={(e) => {setSymbol(e.target.value); setSelected("");}}>{data.tickers.map((value:string)=><option key={value}>{value}</option>)}</select>
            <div className="segmented"><button className={type==="call"?"active":""} onClick={()=>{setType("call");setSelected("");}}>Calls</button><button className={type==="put"?"active":""} onClick={()=>{setType("put");setSelected("");}}>Puts</button></div>
            <select value={effectiveExpiration ?? ""} onChange={(e)=>{setExpiration(e.target.value);setSelected("");}}>{expirations.map((value)=><option key={value}>{value}</option>)}</select>
          </div>
          <div className="table-wrap chain-table"><table><thead><tr><th>Contract</th><th>Strike</th><th>Bid</th><th>Ask</th><th>Spread</th><th>IV</th><th>Delta</th><th>OI</th></tr></thead><tbody>
            {filtered.map((item:Json)=><tr key={item.contractSymbol} className={item.contractSymbol===contract?.contractSymbol?"selected":""} onClick={()=>setSelected(item.contractSymbol)}><td>{item.contractSymbol}</td><td>{money(item.strike)}</td><td>{money(item.bid)}</td><td>{money(item.ask)}</td><td>{money(item.ask-item.bid)}</td><td>{percent(item.impliedVolatility,1)}</td><td>{Number(item.delta).toFixed(3)}</td><td>{Number(item.openInterest).toLocaleString()}</td></tr>)}
          </tbody></table></div>
        </div>
        <aside className="order-ticket">
          <div className="ticket-head"><div><div className="eyebrow">ORDER TICKET</div><h2>{data.symbol} {type.toUpperCase()}</h2></div><span className="state-badge"><StatusDot state={data.session.provider_state}/>{data.session.provider_state}</span></div>
          {!contract ? <EmptyState>No contracts match these controls.</EmptyState> : <>
            <div className="contract-id">{contract.contractSymbol}</div>
            <div className="quote-grid"><div><span>Bid</span><strong>{money(contract.bid)}</strong></div><div><span>Ask</span><strong>{money(contract.ask)}</strong></div></div>
            <dl className="ticket-details"><div><dt>Strike</dt><dd>{money(contract.strike)}</dd></div><div><dt>Expiration</dt><dd>{contract.expiration}</dd></div><div><dt>Delta</dt><dd>{Number(contract.delta).toFixed(3)}</dd></div><div><dt>Vega</dt><dd>{Number(contract.vega).toFixed(3)}</dd></div></dl>
            <label className="quantity-field">Contracts<input type="number" min="1" max="100" value={quantity} onChange={(e)=>setQuantity(Number(e.target.value))}/></label>
            <div className="estimate"><span>Estimated debit</span><strong>{money(quantity*contract.ask*100)}</strong><small>Commission excluded</small></div>
            <button className="button primary wide" onClick={()=>send("buy")}>Paper buy at ask</button>
            <button className="button secondary wide" onClick={()=>send("sell")}>Paper sell at bid</button>
            <p className="ticket-note">Simulated only. Orders use saved bid/ask quotes and never reach a live broker.</p>
          </>}
        </aside>
      </section>
    </>
  );
}

function PortfolioPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/portfolio", refreshKey);
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  return <><PageHeader eyebrow="MANUAL PAPER ACCOUNT" title="Portfolio" description="Liquidation-aware positions, account value, and a complete simulated fill audit trail." />
    <section className="metric-grid four"><Metric label="Cash" value={money(data.account.cash)} /><Metric label="Option value" value={money(data.market_value)} /><Metric label="Total equity" value={money(data.total_equity)} /><Metric label="Realized P&L" value={money((data.trades??[]).reduce((sum:number,item:Json)=>sum+Number(item.realized_pnl),0))} /></section>
    <section className="panel section"><div className="panel-heading"><h2>Open positions</h2><WalletCards size={18}/></div>{!data.positions.length?<EmptyState>No manual paper positions.</EmptyState>:<div className="table-wrap"><table><thead><tr>{Object.keys(data.positions[0]).slice(0,8).map((key)=><th key={key}>{titleCase(key)}</th>)}</tr></thead><tbody>{data.positions.map((item:Json,index:number)=><tr key={index}>{Object.keys(item).slice(0,8).map((key)=><td key={key}>{String(item[key]??"—")}</td>)}</tr>)}</tbody></table></div>}</section>
    <section className="panel section"><div className="panel-heading"><h2>Fill ledger</h2><History size={18}/></div>{!data.trades.length?<EmptyState>No manual paper fills.</EmptyState>:<div className="table-wrap"><table><thead><tr><th>Time</th><th>Side</th><th>Contract</th><th>Quantity</th><th>Price</th><th>Notional</th><th>Realized P&L</th></tr></thead><tbody>{data.trades.map((item:Json)=><tr key={item.id}><td>{item.executed_at.slice(0,19).replace("T"," ")}</td><td>{item.side.toUpperCase()}</td><td>{item.contract_symbol}</td><td>{item.quantity}</td><td>{money(item.price)}</td><td>{money(item.notional)}</td><td>{money(item.realized_pnl)}</td></tr>)}</tbody></table></div>}</section>
  </>;
}

function ResearchPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/research", refreshKey);
  if (loading || !data) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  return <><PageHeader eyebrow="EVIDENCE" title="Research" description="Validation selection, held-out paths, arena provenance, and saved experiment history." />
    <section className="metric-grid four"><Metric label="Saved runs" value={data.run_count} /><Metric label="Arena tickers" value={data.arena.length} /><Metric label="Activated" value={data.arena.filter((item:Json)=>item.Activation==="Active").length} /><Metric label="GNN winners" value={data.arena.filter((item:Json)=>String(item["Selected encoder"]).includes("Graph")).length} /></section>
    <section className="panel section"><div className="panel-heading"><div><h2>Latest arena evidence</h2><p>Independent ticker paths · no portfolio claim</p></div><FlaskConical size={18}/></div>{!data.arena.length?<EmptyState>No arena evidence found.</EmptyState>:<div className="table-wrap"><table><thead><tr><th>Ticker</th><th>Agent</th><th>Encoder</th><th>Activation</th><th>Held-out</th><th>Sandbox</th><th>Drawdown</th><th>Evidence</th></tr></thead><tbody>{data.arena.map((item:Json)=><tr key={item.Ticker}><td><strong>{item.Ticker}</strong></td><td>{item["Selected agent"]}</td><td>{item["Selected encoder"]}</td><td>{item.Activation}</td><td>{percent(item["Held-out return"])}</td><td>{percent(item["Sandbox return"])}</td><td>{percent(item["Max drawdown"])}</td><td>{item.Evidence}</td></tr>)}</tbody></table></div>}</section>
    <section className="panel section"><div className="panel-heading"><h2>Experiment registry</h2><History size={18}/></div><div className="run-grid">{data.runs.slice(0,12).map((run:Json,index:number)=><article className="run-card" key={index}><span>{run.symbol}</span><h3>{run.name}</h3><p>{run.schema_version}</p><small>{run.fold_count} fold(s)</small></article>)}</div></section>
  </>;
}

function SystemPage({ refreshKey }: { refreshKey: number }) {
  const { data, loading, error } = useResource<Json>("/api/overview", refreshKey);
  const jobs = useResource<Json[]>("/api/jobs", refreshKey);
  if (loading || !data || jobs.loading || !jobs.data) return <LoadingState />;
  if (error || jobs.error) return <ErrorState message={error || jobs.error} />;
  return <><PageHeader eyebrow="LOCAL RUNTIME" title="System" description="Application build, service heartbeats, control jobs, and captured process output." />
    <section className="metric-grid four"><Metric label="Application" value={`v${data.version}`} detail="FastAPI + React" /><Metric label="Mode" value="Paper" detail="no live broker adapter" /><Metric label="Data directory" value={`${data.tickers.length} tickers`} detail="saved CSV snapshots" /><Metric label="Control jobs" value={jobs.data.length} detail="current app session" /></section>
    <section className="panel section"><div className="panel-heading"><h2>Job console</h2><ServerCog size={18}/></div><JobStrip jobs={jobs.data}/>{jobs.data[0]?.output?.length>0&&<pre className="log-console">{jobs.data[0].output.join("\n")}</pre>}</section>
  </>;
}

export default function App() {
  const [page, setPage] = useState<Page>("command");
  const [refreshKey, setRefreshKey] = useState(0);
  const overview = useResource<Json>("/api/overview", refreshKey);
  const title = navigation.find((item) => item.id === page)?.label ?? "Control Room";
  const content = useMemo(() => {
    const props = { refreshKey };
    switch (page) {
      case "command": return <CommandCenter {...props} refresh={() => setRefreshKey((value) => value + 1)} />;
      case "agents": return <AgentsPage {...props} />;
      case "models": return <ModelsPage {...props} />;
      case "training": return <TrainingPage {...props} />;
      case "trade": return <TradePage {...props} />;
      case "portfolio": return <PortfolioPage {...props} />;
      case "research": return <ResearchPage {...props} />;
      case "system": return <SystemPage {...props} />;
    }
  }, [page, refreshKey]);
  const services = overview.data?.services ?? [];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={21} /></div>
          <div><strong>OPTIONS LAB</strong><span>CONTROL ROOM</span></div>
        </div>
        <nav>
          <div className="nav-label">Workspace</div>
          {navigation.map(({ id, label, icon: Icon }) => (
            <button className={page === id ? "nav-item active" : "nav-item"} onClick={() => setPage(id)} key={id}>
              <Icon size={18} /><span>{label}</span>{id === "training" && <small>NEW</small>}
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <div className="nav-label">Runtime</div>
          {services.map((service: Json) => <div key={service.id}><StatusDot state={service.status}/><span>{service.label}</span></div>)}
        </div>
        <div className="paper-lock"><ShieldCheck size={17}/><div><strong>Paper environment</strong><span>No live broker connection</span></div></div>
      </aside>
      <div className="workspace">
        <div className="topbar">
          <div className="breadcrumb"><span>Options Lab</span><ChevronRight size={14}/><strong>{title}</strong></div>
          <div className="topbar-state">
            <span className="environment-badge">PAPER</span>
            <span><StatusDot state={overview.data?.market?.session?.provider_state ?? "unknown"}/>{overview.data?.market?.session?.provider_state ?? "Market unknown"}</span>
            <span className="clock">{new Date().toLocaleDateString("en-US",{month:"short",day:"numeric",year:"numeric"})}</span>
            <button className="top-refresh" onClick={() => setRefreshKey((value) => value + 1)} title="Refresh"><RefreshCw size={16}/></button>
          </div>
        </div>
        <main className="content">{content}</main>
      </div>
    </div>
  );
}
