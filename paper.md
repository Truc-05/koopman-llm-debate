
\section{Related Work}
\label{sec:related}
\textbf{Multi-agent LLM debate.} Introduced by \cite{ref1}, MAD improves reasoning by iterated propose--critique--revise \cite{ref3}. Subsequent work targets the \emph{protocol}: divergence delays premature convergence \cite{ref2}; gains are prompt/aggregation-sensitive \cite{ref4}; \cite{ref17} attribute most benefit to voting alone; \cite{ref5,ref6} document echo-chamber regimes where accuracy \emph{decreases}. None identifies a \emph{controlled} transition operator.

\textbf{Opinion dynamics.} DeGroot \cite{ref7} sets $x_{t+1}=Wx_t$; FJ \cite{ref8} adds prejudice, $x_{t+1}=\lambda Wx_t+(1-\lambda)x_0$; both are LLM-opinion baselines \cite{ref18}. The 2026 works \cite{ref9,ref10} fit data yet stay linear and autonomous, spectrally analyzing a hand-specified operator with no control input; ours subsumes them while identifying $B$, and our response-collapse finding further explains \emph{why} their fixed static form suffices. A separate strand \emph{simulates} opinion dynamics with LLMs \cite{ref19}.

\textbf{Koopman control.} The Koopman operator \cite{ref20} lifts a nonlinear system to a linear one on observables, estimated by EDMD \cite{ref11} with SVD variants \cite{ref21}; actuated extensions \cite{ref12,ref22} enable MPC/LQR. Applications span fluids, power, and robot swarms \cite{ref16,ref15} and agent-based \emph{social} dynamics \cite{refNiemann}. Koopman/EDMD and dynamic mode decomposition have recently been applied to a \emph{single} LLM (hallucination detection) \cite{refWilsonAkrout,refAkrout}, but passively and without a multi-agent or control setting; no prior work identifies a \emph{controlled} Koopman operator for a \emph{multi-agent} LLM system. Table~\ref{tab:related} positions us.

\begin{table}[t]
\centering
\caption{Positioning relative to prior work.}
\label{tab:related}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\columnwidth}{@{}Xcccc@{}}
\toprule
Model & Data-driven & Nonlinear & Identified & Control \\
\midrule
DeGroot \cite{ref7} & \ding{55} & \ding{55} & \ding{55} & \ding{55} \\
Friedkin--Johnsen \cite{ref8} & \ding{55} & \ding{55} & \ding{55} & \ding{55} \\
Hidden Anchors \cite{ref9} & partial & \ding{55} & \ding{55} & \ding{55} \\
Delayed Verif.\ \cite{ref10} & \ding{55} & \ding{55} & \ding{55} & \ding{55} \\
Koopman-LLM (single) \cite{refWilsonAkrout} & \ding{51} & \ding{51} & \ding{51} & \ding{55} \\
Koopman-MAS \cite{ref15} & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
\textbf{EigenDebate (ours)} & \ding{51} & \ding{51} & \ding{51} & \ding{51} \\
\bottomrule
\end{tabularx}
\end{table}

\section{Debate as a Controlled Koopman System}
\label{sec:method}

\subsection{Setup}
\label{sec:setup}
Agent $i$ holds logits $x^i_t\in\mathbb R^K$ over the $K$ answers (with unknown correct answer $a^\star$); the joint state
\begin{equation}
z_t=(x^1_t,\dots,x^N_t)\in\mathbb R^{d},\qquad d=NK,
\end{equation}
evolves by one debate round \eqref{eq:intro-dynamics}, and a control input $u_t\in\mathbb R^{m}$ encodes an intervention---a one-hot push toward an answer, together with its timing and intensity. Debate is thus a \emph{controlled Markov process} with kernel $P_u(z,\cdot)=\Pr[T(z,u,\xi)\in\cdot]$, under mild regularity (Markovianity in $z_t$, an $L^2$ reference measure, per-family stationarity; detailed in the supplement). Its stakes live in two observables,
\begin{equation}
\bar p(z)=\tfrac1N\textstyle\sum_{i}\mathrm{softmax}(x^i),\qquad m_x(z)=\bar p(z)_x ,
\label{eq:observables}
\end{equation}
the mean belief and the \emph{steering margin} toward answer $x$: an adversary drives $m_{x_{\mathrm{adv}}}$ up for a wrong $x_{\mathrm{adv}}\ne a^\star$, and moderation is the exact dual.

\subsection{The controlled Koopman operator}
\label{sec:koopman}
Rather than model the intractable $T$, we model how \emph{expectations of observables} propagate under it.
\begin{definition}[controlled stochastic Koopman operator]
\label{def:koopman}
For a fixed input $u$, the operator $\mathcal K_u:L^2(\mu)\to L^2(\mu)$ acts on observables by
\begin{equation}
(\mathcal K_u g)(z)=\mathbb E\big[g(z_{t+1})\mid z_t=z,\,u_t=u\big].
\label{eq:koopman-def}
\end{equation}
\end{definition}
The payoff is exact linearity: $\mathcal K_u$ is a \emph{linear} operator no matter how nonlinear or stochastic $T$ is, because conditional expectation is linear in the observable $g$---at the cost of infinite dimension, which we tame by projecting onto a finite dictionary $\Psi(z)=[1,z,\dots]^\top$ that contains the state block ($z=P_z\Psi$). Linearizing the input then turns \eqref{eq:koopman-def} into the identified state-space model \eqref{eq:edmdc-model}, $\Psi(z_{t+1})\approx K\Psi(z_t)+Bu_t$. Two structural facts drive everything downstream. First, for an invariant sampling measure Jensen's inequality gives $\|\mathcal K\|\le1$, so the autonomous spectrum is confined to the unit disk,
\begin{equation}
\sigma(\mathcal K)\subseteq\{\lambda:|\lambda|\le1\},
\label{eq:unit-disk}
\end{equation}
a stability statement that doubles as an estimation diagnostic (any identified $|\hat\lambda|>1$ is noise). Second, diagonalizing $K=V\Lambda V^{-1}$ and writing $\phi=V^{-1}\Psi$, every expected measurement is a \emph{superposition of Koopman modes},
\begin{equation}
\phi_{t}=\Lambda^{t}\phi_0+\sum_{k=0}^{t-1}\Lambda^{t-1-k}(V^{-1}B)\,u_k ,
\label{eq:mode}
\end{equation}
in which $|\lambda_i|<1$ modes decay geometrically, $\lambda_i=1$ modes are the invariants that fix the consensus the uncontrolled debate drifts into, and the rows of $V^{-1}B$ say how an input reaches each mode (derivations in the supplement).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{koopman_spectrum.png}
\caption{Koopman spectrum. Eigenvalues near the unit circle are persistent consensus modes; smaller magnitudes decay geometrically. The empirical spectral radius is stable throughout ($\rho(A)<1$): $0.62$--$0.82$ on Qwen, Mistral, and Llama, and near-unit-root ($0.96$--$1.00$) on Gemma, whose beliefs are near-static.}
\label{fig:spectrum}
\end{figure}

\begin{proposition}[autonomous consensus]
\label{prop:consensus}
If the autonomous chain $P_0$ has absorbing consensus classes $C_1,\dots,C_r$ and is absorbed w.p.\ $1$, then $E_1=\ker(\mathcal K-I)$ has $\dim E_1=r$ with basis the absorption probabilities $q_j(z)=\Pr[z_t\to C_j\mid z_0=z]$, and for every $g\in L^2(\mu)$,
\begin{equation}
\lim_{t\to\infty}\mathbb E[g(z_t)\mid z_0]=\sum_{j=1}^{r} q_j(z_0)\,\mathbb E_{\pi_j}[g];
\label{eq:consensus-limit}
\end{equation}
the limit is unique and $z_0$-independent iff $r=1$ and $1$ is the only unimodular eigenvalue.
\end{proposition}
Proposition~\ref{prop:consensus} (proof in App.~\ref{app:proofs}) characterizes the attractor the \emph{uncontrolled} debate drifts into, against which every intervention acts.

\subsection{Data-driven estimation}
\label{sec:estimator}
We identify $(K,B)$ from $R$ debate logs by stacking transitions $(\Psi_X,\Psi_Y)$ and inputs $U$ and solving a ridge least squares,
\begin{equation}
[\widehat K\ \ \widehat B]=\arg\min_{K,B}\ \|\Psi_Y-K\Psi_X-BU\|_F^2+\gamma\big\|[K\ \ B]\big\|_F^2 .
\label{eq:edmdc}
\end{equation}
This is a Galerkin projection of $\mathcal K$: because $\mathcal K_u$ is a conditional expectation \eqref{eq:koopman-def}, the LLM's sampling noise enters as a \emph{zero-mean residual, not a bias}, and $\widehat K$ converges to the true spectrum as data and dictionary grow. Three conventions keep the downstream spectral and control quantities correct---taking \emph{left} eigenvectors as the Koopman eigenfunctions (verified on held-out pairs), projecting any spurious $|\hat\lambda|>1$ back into the unit disk \eqref{eq:unit-disk}, and bootstrapping over \emph{debates} rather than transitions, the exchangeable unit. The closed-form solution, the spectral projection, and the full identification algorithm are given in the supplement.

\section{Control-Theoretic Characterization}
\label{sec:characterization}
Linearity makes the model \emph{analyzable}: from the state block $A=P_z\widehat K P_z^\top$ and input block $B=P_z\widehat B$ the control anatomy follows in closed form.

\begin{proposition}[stability and consensus time]
\label{prop:stability}
If $\rho(A)=\max_i|\lambda_i(A)|<1$, the autonomous debate is asymptotically stable: there is $P\succ0$ with $A^\top PA-P\prec0$, and
\begin{equation}
\begin{aligned}
\|\mathbb E[z_t\mid z_0]-z_\infty\|&\le C(z_0)\,\rho(A)^t,\\
t_\varepsilon(z_0)&\le\frac{\log(C(z_0)/\varepsilon)}{\log(1/\rho(A))}.
\end{aligned}
\label{eq:consensus-time}
\end{equation}
The spectral gap $1-\rho(A)$ thus sets the window in which an intervention can still act before consensus locks in.
\end{proposition}

\begin{figure}[t]
\centering
\includegraphics[width=1.0\columnwidth]{controllability.png}
\caption{Control-theoretic characterization. The controllability matrix, finite-horizon Gramian $W_H$, and reachable ellipsoid $\mathcal R_H$ (Theorem~\ref{thm:controllability}) quantify how far a moderator or adversary can steer the debate; strong anisotropy ($\kappa\gg1$) makes some wrong consensuses freely installable, others near-unreachable.}
\label{fig:controllability}
\end{figure}

\begin{theorem}[controllability and reachable set]
\label{thm:controllability}
Let $\mathcal C_H=[B,AB,\dots,A^{H-1}B]$ and
\begin{equation}
W_H=\sum_{k=0}^{H-1}A^{k}BB^\top (A^\top)^{k}\succeq0 .
\label{eq:gramian}
\end{equation}
Then (i) the debate is controllable iff $\operatorname{rank}\mathcal C_d=d$ (equivalently $W_H\succ0$); (ii) the set reachable in $H$ rounds under $\sum_h\|u_h\|^2\le1$ is the ellipsoid
\begin{equation}
\mathcal R_H=\{W_H^{1/2}v:\|v\|\le1\},
\label{eq:reach}
\end{equation}
with semi-axes $\sqrt{\lambda_i(W_H)}$; (iii) the belief mass an adversary installs on target $x$ from a uniform prior is $\mathrm{reach}(x)=m_x(P_z\psi_H)$, $\psi_{k+1}=\widehat K\psi_k+\widehat Be_x$; and (iv) the minimum-energy input to any $\zeta\in\mathcal R_H$ is
\begin{equation}
u^\star=\mathcal C_H^\top(\mathcal C_H\mathcal C_H^\top)^{-1}\zeta,\qquad \|u^\star\|^2=\zeta^\top W_H^{-1}\zeta,
\label{eq:min-energy}
\end{equation}
cheap along the top eigenvectors of $W_H$, prohibitive along its near-null directions.
\end{theorem}

\begin{definition}[manipulability and resilience margin]
\label{def:manip}
The control anisotropy is $\kappa(W_H)=\lambda_{\max}(W_H)/\lambda_{\min}(W_H)$; against a single Byzantine agent the resilience margin is
\begin{equation}
\varrho=1-\max_{x\ne a^\star}\mathrm{reach}(x)\in[0,1],
\label{eq:resilience}
\end{equation}
computed from $(A,B)$ without an exhaustive attack. Empirically (Sec.~\ref{sec:experiments}) debate is stable and controllable in rank yet strongly anisotropic ($\kappa\gg1$), so $\varrho$ is answer-dependent.
\end{definition}

Algorithm~\ref{alg:audit} computes this audit. We now record a structural property that governs the rest of the paper: the operator is blind to the truth.

\begin{algorithm}[t]
\caption{Control-theoretic audit of the identified operator.}
\label{alg:audit}
\begin{algorithmic}[1]
\Require blocks $(A,B)$; horizon $H$; uniform prior $z_{\mathrm u}$; correct answer $a^\star$
\Ensure $\rho$; controllability rank; $\sigma(W_H)$; $\kappa$; $\mathrm{reach}(\cdot)$; $\varrho$; $\ker M_H$
\State $\rho\gets\max_i|\lambda_i(A)|$; \textbf{assert} $\rho<1$ \Comment{Prop.~\ref{prop:stability}}
\State $\mathcal C_H\gets[B,AB,\dots,A^{H-1}B]$; $\mathrm{rank}\gets\operatorname{rank}\mathcal C_H$ \Comment{Thm.~\ref{thm:controllability}(i)}
\State $W_H\gets\sum_k A^kBB^\top(A^\top)^k$; $\kappa\gets\lambda_{\max}(W_H)/\lambda_{\min}(W_H)$
\For{$x\in\mathcal A$}
  \State $\psi\gets\Psi(z_{\mathrm u})$; repeat $H\times$: $\psi\gets\widehat K\psi+\widehat B e_x$; $\mathrm{reach}(x)\gets m_x(P_z\psi)$
\EndFor
\State $\varrho\gets1-\max_{x\ne a^\star}\mathrm{reach}(x)$ \Comment{Def.~\ref{def:manip}}
\State $M_H\gets\sum_k(A^\top)^kC^\top CA^k$; report $\operatorname{rank}\mathcal O_H,\ \kappa(M_H),\ \ker M_H$
\State trajectory-bootstrap $\{\rho,\sigma(W_H),\mathrm{reach}(\cdot),\varrho\}$ for CIs
\end{algorithmic}
\end{algorithm}

\begin{proposition}[truth-agnosticity]
\label{prop:truth}
The identified dynamics $(K,B)$ are independent of $a^\star$: the input matrix acts identically on every target, $\widehat Be_x$ sharing the same law across all $x\in\mathcal A$, so the reachability rollout of Theorem~\ref{thm:controllability}(iii) is one operator for every target. The truth enters only through the outer selection in \eqref{eq:resilience} and the accuracy read-out, never through the dynamics.
\end{proposition}

Steering toward truth and toward falsehood are thus the \emph{same} operation on different targets: the debate carries \emph{steering authority} but no \emph{epistemic content}. Two consequences follow.

\begin{corollary}[attack--defense duality]
\label{cor:duality}
Manipulation and moderation are a single control channel with opposite targets--an adversary applies $u_t=e_{x_{\mathrm{adv}}}$, a moderator $u_t=e_c$ toward an honest target $c$--both through the identical $B$. A label-free defense suppressing $m_{x_{\mathrm{adv}}}$ without access to $a^\star$ is therefore available whenever moderation is, its authority bounded by the same reachability \eqref{eq:reach}.
\end{corollary}

Empirically (Sec.~\ref{sec:experiments}) a single adversary installs a wrong consensus in up to $48\%$ of Qwen debates ($56\%$ on Gemma, $60\%$ on Mistral); a label-free honest-margin controller on the same channel cuts the attack-success rate to $\le2\%$ (Qwen; $\le13\%$ on Mistral, Llama, and Gemma) and recovers accuracy ($+14$ pts), naming the true answer as its target in only $18$--$44\%$ of cases (near the $25\%$ chance rate for four options)--suppression, not truth-injection, as Proposition~\ref{prop:truth} predicts.

\begin{remark}[unmoderated debate reduces to voting]
\label{rem:choi}
Proposition~\ref{prop:truth}, with near-persistence ($\rho(A)<1$, slow drift), accounts for the finding of \cite{ref17} that most of debate's benefit is attributable to aggregation alone: absent a control input, deliberation adds no epistemic content beyond a static vote, so its value is realized only under active steering.
\end{remark}

\section{Controllability versus Observability}
\label{sec:contr-obs}
Theorem~\ref{thm:controllability} bounds what \emph{some} intervention achieves; it is silent on whether that intervention can be \emph{chosen} from what a moderator observes. We separate the two by a decision-theoretic analogue of the Kalman decomposition. The \textbf{oracle} selects, using the realized outcome,
\begin{equation}
u_t^{\mathrm{orc}}=\arg\max_{u\in\mathcal U}\ \mathbb E[m_{a^\star}(z_{t+1})\mid z_t,u],
\label{eq:oracle}
\end{equation}
while an \textbf{observation-based} controller acts through $\pi(\Psi(z_t))$. With $\mathrm{Acc}(\cdot)$ the induced accuracy, define the headroom $\mathcal H=\mathrm{Acc}(u^{\mathrm{orc}})-\mathrm{Acc}(\varnothing)$ and the controllability--observability gap
\begin{equation}
\Delta_{\mathrm{co}}=\mathrm{Acc}(u^{\mathrm{orc}})-\!\!\sup_{\pi\ \Psi\text{-measurable}}\!\!\mathrm{Acc}(\pi(\Psi(z)))\ \ge0 .
\label{eq:co-gap}
\end{equation}
Equip the system with the read-out $y_t=C\Psi(z_t)$ ($C=P_z$), observability matrix and Gramian
\begin{equation}
\begin{aligned}
\mathcal O_H&=\big[\,C^\top,\,(CA)^\top,\,\dots,\,(CA^{H-1})^\top\,\big]^\top,\\
M_H&=\sum_{k=0}^{H-1}(A^\top)^k\,C^\top C\,A^k .
\end{aligned}
\label{eq:obsv}
\end{equation}
Classically the residual $\Delta_{\mathrm{co}}$ would live in $\operatorname{Im}\mathcal C_d\cap\ker\mathcal O_H$. Empirically, however, the identified system is \emph{classically observable} on every benchmark--$\mathcal O_H$ full rank, $\ker M_H=\{0\}$, and $M_H$ no worse conditioned than the controllability Gramian $W_H$--so that subspace is trivial and cannot host $\Delta_{\mathrm{co}}$. The residual is therefore \emph{epistemic}: we distinguish \emph{state observability} (the trajectory pins down the latent state; here complete) from \emph{epistemic observability} (the state pins down the accuracy-optimal action; here incomplete, since by Proposition~\ref{prop:truth} the truth is exogenous).

\begin{proposition}[when the gap closes]
\label{prop:gap}
$\Delta_{\mathrm{co}}=0$ iff the oracle input \eqref{eq:oracle} is $\sigma(\Psi(z))$-measurable; any component of the optimal action depending on outcome information not encoded in $\Psi(z_t)$ contributes a strictly positive term to $\Delta_{\mathrm{co}}$.
\end{proposition}

Empirically the headroom $\mathcal H$ is significant while every observation-based policy--learned regressors and a myopic heuristic alike--plateaus at the same level, so $\Delta_{\mathrm{co}}>0$. Whether this is a fundamental epistemic-observability limit or a finite-data effect is undetermined; we pose \eqref{eq:co-gap} as the central open problem and give the first evidence, not a claimed resolution.

\section{Control Synthesis and Generalization}
\label{sec:mpc}
Because \eqref{eq:edmdc-model} is linear in observable space, it supplies standard synthesis. For a target lift $\psi^\star$, the moderator solves the receding-horizon program
\begin{equation}
\begin{aligned}
\min_{u_0,\dots,u_{H-1}}\ &\sum_{h=1}^{H}\|\psi_h-\psi^\star\|_Q^2+\sum_{h=0}^{H-1}\|u_h\|_R^2\\
\text{s.t.}\ &\psi_{h+1}=\widehat K\psi_h+\widehat B u_h,\ \psi_0=\Psi(z_t),
\end{aligned}
\label{eq:mpc}
\end{equation}
with stationary LQR policy $u_t=-G(\psi_t-\psi^\star)$, $G=(R+\widehat B^\top P\widehat B)^{-1}\widehat B^\top P\widehat K$, where $P$ solves the discrete Riccati equation
\begin{equation}
P=\widehat K^\top P\widehat K-\widehat K^\top P\widehat B(R+\widehat B^\top P\widehat B)^{-1}\widehat B^\top P\widehat K+Q .
\label{eq:dare}
\end{equation}
In eigen-coordinates \eqref{eq:mode} control is mode-wise: mode $i$ is reachable iff $(WB)_{i:}\ne0$, the PBH test $\operatorname{rank}[A-\lambda_iI,\,B]=d$. This \emph{designs} interventions (Alg.~\ref{alg:mpc}); whether the closed loop attains the oracle is exactly the open gap $\Delta_{\mathrm{co}}$.

\begin{algorithm}[t]
\caption{Koopman-MPC moderation.}
\label{alg:mpc}
\begin{algorithmic}[1]
\Require $(\widehat K,\widehat B)$; target $\psi^\star$; horizon $H$; weights $Q,R$; feasible set $\mathcal U_{\mathrm{feas}}$
\Loop\ each round $t$
\State $\psi\gets\Psi(z_t)$; solve QP \eqref{eq:mpc} (or LQR gain \eqref{eq:dare}) for $u_0^\star$
\State apply $u\gets\operatorname{Proj}_{\mathcal U_{\mathrm{feas}}}(u_0^\star)$; observe $z_{t+1}$
\EndLoop
\end{algorithmic}
\end{algorithm}

\textbf{Generalization.} The framework strictly contains fixed-form models. If $T$ is affine, $z_{t+1}=Mz_t$, then for any linear observable $g_c(z)=c^\top z$,
\begin{equation}
(\mathcal K g_c)(z)=c^\top Mz=g_{M^\top c}(z),
\label{eq:generalize}
\end{equation}
so the degree-one compression of $\mathcal K$ \emph{is} $M^\top$: EDMD reduces to least-squares identification of $M$, Proposition~\ref{prop:stability} recovers DeGroot/FJ convergence, and the baselines are nested ablations (degree-1 EDMD $\supset$ FJ $\supset$ DeGroot) that lack the input matrix $B$ on which \eqref{eq:resilience} depends.

\section{Experiments}
\label{sec:experiments}
We instantiate the framework as a single pipeline--\emph{identify} $(K,B)$, \emph{audit} it in closed form, then \emph{attack} and \emph{defend} through the same channel--and ask whether that pipeline behaves consistently across model families. The evaluation grid is four models $\times$ four strong four-way benchmarks (MMLU-math, MATH, TruthfulQA, ARC-Challenge), i.e.\ $16$ identification cells, each built from $60$ collected trajectories for identification and a disjoint $n=90$ for evaluation ($\approx2.4$k debates, $\approx15$k belief transitions); large-$K$ BBH-logic ($K\in\{5,7\}$) enters only as a boundary case. Every quantity below is recomputed from committed logs (including the digital-twin column, refit from scratch), and all confidence bands use the trajectory bootstrap of Sec.~\ref{sec:estimator}.
\subsection{Setup and protocol}
Each debate runs $N=4$ agents for $T=6$ rounds on a multiple-choice question; the belief state $z_t\in\mathbb R^{NK}$ stacks the agents' per-answer scores. One agent is adversarial and argues a fixed wrong answer $x_{\mathrm{adv}}$. From round $k_{\mathrm{def}}=2$ a defender may inject a single one-hot push toward one answer--the \emph{same} input channel $B$ the attacker uses. We identify $(K,B)$ per cell by degree-one EDMDc \eqref{eq:edmdc} with ridge $\gamma=10^{-6}$. The defender optimizes the \emph{label-free} honest margin $m_{c_{\mathrm{hon}}}-m_{x_{\mathrm{adv}}}$; we compare three policies: no defense, the \emph{digital-twin} MPC that plans the push on $(K,B)$ alone \eqref{eq:edmdc-model} (observation-based, Alg.~\ref{alg:mpc}), and the \emph{oracle} \eqref{eq:oracle} that selects the realized-best push. Qwen2.5-7B is the main model; Mistral-7B, Llama-3.1-8B and Gemma-2-9B are replications; all are run locally.
\subsection{Identification and one-step fidelity}
The estimand is the input channel $B$, not the autonomous forecast. As anticipated in Sec.~\ref{sec:intro}, beliefs are quasi-static after early convergence, so persistence is a strong one-step predictor and we do \emph{not} claim to beat it. What we test (Table~\ref{tab:fid}) is whether adding the control channel to the autonomous model \eqref{eq:generalize} captures real input-driven variation: on held-out control-horizon belief error the identified $(K,B)$ improves over the autonomous DeGroot/FJ model on $5/7$ Qwen benchmarks and on $11$ of $12$ replication cells (all eight Llama and Gemma cells, three of four Mistral), confirming $B$ is identifiable and non-trivial. A higher-capacity nonlinear (MLP) fit is uniformly \emph{worse} on held-out rollout--overfitting at this per-cell sample size--and its gap to $(K,B)$ is within the bootstrap band on the cell where they are closest, consistent with a near-linear controlled response at degree one. One-step fidelity is thus a sanity check, not the operative validation of $(K,B)$; the control-theoretic audit and the oracle/defense outcomes are.
\subsection{Spectral structure and stability}
Diagonalizing each $A=P_z\widehat K P_z^\top$ (Fig.~\ref{fig:spectrum}) confirms the operator-norm bound \eqref{eq:unit-disk} empirically: every eigenvalue lies inside the unit disk, so by Prop.~\ref{prop:stability} the autonomous debate is asymptotically stable and drifts to consensus. The contraction rate is model-dependent (Table~\ref{tab:struct}): the spectral radius is $\rho(A)=0.62$--$0.82$ on Qwen, Mistral and Llama versus $0.96$--$1.00$ on Gemma. The spectral gap $1-\rho(A)$ therefore sets the intervention window of Prop.~\ref{prop:stability}: it is a comfortable $0.2$--$0.4$ on the first three models but only $\le0.04$ on Gemma, whose near-unit-root dynamics barely contract--matching both its more static beliefs and, as we will see, its smaller control headroom.
\subsection{Controllability, reachability, and manipulability}
The controllability matrix is full rank on \emph{every} cell (rank\,$\mathcal C_d=d$, $d=16$ at $K=4$ up to $28$ at $K=7$; Table~\ref{tab:struct}), so by Thm.~\ref{thm:controllability}(i) deliberation is controllable: some input sequence drives the belief state anywhere. But the finite-horizon Gramian $W_H$ is strongly anisotropic ($\log_{10}\kappa(W_H)\approx5.7$--$7.2$; Fig.~\ref{fig:controllability}), so the reachable ellipsoid \eqref{eq:reach} is a needle--its longest semi-axis exceeds its shortest by roughly three orders of magnitude--and the min-energy input \eqref{eq:min-energy} is cheap along a few consensus directions and prohibitive along the rest. Instantiated as the per-answer manipulability of Thm.~\ref{thm:controllability}(iii) (Table~\ref{tab:reach}, Fig.~\ref{fig:reach}), most targets are installable from a uniform prior (reach $\approx1$; mean manipulability index $0.53$--$0.96$) while a minority stay near-unreachable (down to $0.04$). Because the collected push counts are balanced across answers, these low-reach targets are a genuine positional bias of the model, not an identification artifact--a certifiable, answer-specific vulnerability that autonomous opinion-dynamics models cannot even express. The model-based twin-MPC that plans on this $(K,B)$ is a valid controller (Alg.~\ref{alg:mpc}) but, as Sec.~\ref{sec:contr-obs} anticipated, does not reach the oracle; realizing the full reachable set from observation alone is the open control-design problem below.
\subsection{Observability and the control--observation gap}
The dual audit is sharper still. The system is classically \emph{observable} on every cell ($\ker M_H=\{0\}$, $\mathcal O_H$ full rank; Table~\ref{tab:struct}): the mean-belief readout sees $K$ coordinates directly and the dynamics expose the remaining $NK-K$, reconstructing the full per-agent state despite agent pooling. Moreover the observability Gramian is, if anything, \emph{better} conditioned than the controllability one ($\log_{10}\kappa(M_H)\approx4.2$--$6.5$ versus $5.7$--$7.2$, on $17$ of $19$ cells), so the residual below is not a state-observability or conditioning deficiency. Yet the headroom is not attainable from observation: the twin-MPC and a myopic heuristic--both $\Psi(z_t)$-measurable--plateau well below the oracle (on Qwen-MMLU the twin reaches only $0.49$ against the oracle's $0.60$, from a no-defense $0.43$; Table~\ref{tab:main}, Fig.~\ref{fig:accuracy}), so the controllability--observability gap $\Delta_{\mathrm{co}}$ of \eqref{eq:co-gap} is strictly positive in every cell (Prop.~\ref{prop:gap}). By the state/epistemic distinction of Sec.~\ref{sec:contr-obs} this plateau is \emph{epistemic}: the state is pinned down, but the accuracy-optimal action depends on the exogenous truth (Prop.~\ref{prop:truth}) that $\Psi(z_t)$ does not encode. Whether this is fundamental or a finite-sample effect is undetermined; we report it as the central open problem, with first evidence, not a resolution.
\subsection{Manipulation and label-free defense}
Corollary~\ref{cor:duality} makes attack and moderation one channel with opposite targets, and the outcomes bear it out (Table~\ref{tab:asr}, Fig.~\ref{fig:asr}). Left uncontrolled, a single adversary drives the final consensus to its target in up to $48\%$ of Qwen debates ($56\%$ on Gemma, $60\%$ on Mistral). Routing the same $B$ to the label-free honest-margin objective collapses this attack-success rate to $\le2\%$ on Qwen and $\le13\%$ on Mistral, Llama and Gemma--a resilience $1-\mathrm{ASR}\ge0.87$ everywhere. Consistent with Prop.~\ref{prop:truth}, the defender \emph{suppresses} the adversary rather than injecting truth: its push coincides with the true answer in only $18$--$44\%$ of cases (near the $25\%$ chance rate), yet accuracy still recovers because removing the adversarial distortion suffices. That the very controllability enabling the attack also enables the defense--through the identical channel--is the paper's operational point.
\subsection{Accuracy: oracle headroom and generalization}
The oracle \eqref{eq:oracle} quantifies the channel's headroom (Table~\ref{tab:main}, Fig.~\ref{fig:accuracy}), and the pattern replicates across all four models. On Qwen it raises accuracy from $0.43$ to $0.60$ on MMLU-math (McNemar $p=0.001$) with every strong benchmark individually significant (MATH $p=.007$, TruthfulQA $p=.019$, ARC $p=.012$; pooled $+14$ points, CI $[+10,+19]$). Mistral independently reproduces the \emph{full} pattern--all four benchmarks significant (MMLU, MATH $p<.001$; ARC $p=.001$; TruthfulQA $p=.031$) with the largest gains of any model (pooled $+19$, CI $[+15,+24]$)--so two of four models show the effect on every benchmark. Llama reproduces it partially (pooled $+13$, CI $[+9,+18]$; $3/4$ significant, ARC the near-ceiling exception) and Gemma is weakest (pooled $+9$, CI $[+5,+13]$; MMLU and ARC significant, MATH/TruthfulQA not at $n=90$). This label-free gain depends on the \emph{objective}, not merely on intervening: the ablation of Table~\ref{tab:abl} shows the older ``suppress-adversary'' objective ($-m_{x_{\mathrm{adv}}}$) is near-zero or negative on $5/16$ cells, whereas the honest-margin objective is positive on all $16$. The cross-model spread is orderly and points to a headroom account (Fig.~\ref{fig:headroom}): the largest oracle gain is on Mistral, whose low base accuracy leaves most to recover, and the smallest on Gemma, which is both already accurate and barely contracting ($\rho\approx1$); across the $16$ cells the gain trends down with $\rho(A)$ (Pearson $r=-0.51$, which we read as suggestive rather than conclusive since cells within a model share an operator), so the effect attenuates with available headroom rather than vanishing--a controllability, not an inflation, signature. The boundary cases delimit the claim: BBH neutralizes the attack into an honest-side ceiling and its large $K$ thins each cell (BBH-logic at $K=7$: $0.22\to0.27$, $p=.45$), while ARC is near-ceiling with little attack surface. Finally, the manipulability certificate $\varrho$ is computable with bootstrap CIs but only weakly rank-correlated with realized reach (Spearman $\approx0.4$); we report it as a cautionary negative.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]]{fig_accuracy.png}
\caption{Accuracy under no defense, the observation-based digital-twin MPC, and the oracle (honest-margin objective, $n=90$/cell; $*$: McNemar $p<.05$ vs no-def). The twin recovers part of the headroom but plateaus below the oracle in every cell--the epistemic control--observation gap of Sec.~\ref{sec:contr-obs}.}
\label{fig:accuracy}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_asr.png}
\caption{Attack--defense duality (Cor.~\ref{cor:duality}). A single adversary installs a wrong consensus in up to $60\%$ of debates; routing the identical channel $B$ to the label-free honest-margin objective collapses the attack-success rate to $\le13\%$ ($\le2\%$ on Qwen).}
\label{fig:asr}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{fig_reach.png}
\caption{Per-answer reachability $\mathrm{reach}(x)$ from a uniform prior (Thm.~\ref{thm:controllability}(iii)). Most answers are freely installable ($\approx1$) while a minority are near-unreachable ($\le0.1$): a certifiable, answer-specific manipulability and a genuine positional bias.}
\label{fig:reach}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.82\columnwidth]{fig_headroom.png}
\caption{Oracle gain $\Delta$ versus spectral radius $\rho(A)$ across the $16$ cells. The gain trends down with $\rho(A)$ (Pearson $r=-0.51$): near-unit-root operators (Gemma) leave the control channel least headroom. Suggestive, since cells within a model share an operator.}
\label{fig:headroom}
\end{figure}

\begin{table}[t]
\centering
\caption{Accuracy under control (honest-margin objective, $n=90$/cell). ``twin'' is the observation-based digital-twin MPC; $\Delta=$ oracle$-$no-def with trajectory-bootstrap $95\%$ CI; $p$ is McNemar's exact test. Per-model pooled rows in bold. Two of four models are significant on all four benchmarks.}
\label{tab:main}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccl@{\ }c}
\hline
Model & Bench & no-def & twin & oracle & $\Delta$ [95\% CI] & $p$ \\
\hline
Qwen2.5-7B & MMLU & .43 & .49 & .60 & $+.17$\,[$+.08,+.26$] & .001 \\
 & MATH & .29 & .41 & .43 & $+.14$\,[$+.06,+.24$] & .007 \\
 & TQA & .30 & .36 & .42 & $+.12$\,[$+.03,+.21$] & .019 \\
 & ARC & .63 & .77 & .77 & $+.13$\,[$+.04,+.22$] & .012 \\
 & \textbf{pooled} & & & & $\mathbf{+.14}$\,[$+.10,+.19$] & \\
\hline
Mistral-7B & MMLU & .17 & .29 & .37 & $+.20$\,[$+.11,+.30$] & $<$.001 \\
 & MATH & .10 & .31 & .38 & $+.28$\,[$+.18,+.38$] & $<$.001 \\
 & TQA & .10 & .16 & .21 & $+.11$\,[$+.02,+.20$] & .031 \\
 & ARC & .33 & .43 & .52 & $+.19$\,[$+.09,+.29$] & .001 \\
 & \textbf{pooled} & & & & $\mathbf{+.19}$\,[$+.15,+.24$] & \\
\hline
Llama-3.1-8B & MMLU & .28 & .37 & .41 & $+.13$\,[$+.06,+.21$] & .004 \\
 & MATH & .16 & .36 & .34 & $+.19$\,[$+.09,+.29$] & .001 \\
 & TQA & .27 & .40 & .42 & $+.16$\,[$+.07,+.24$] & .001 \\
 & ARC & .57 & .60 & .62 & $+.06$\,[$-.03,+.14$] & .359 \\
 & \textbf{pooled} & & & & $\mathbf{+.13}$\,[$+.09,+.18$] & \\
\hline
Gemma-2-9B & MMLU & .38 & .49 & .52 & $+.14$\,[$+.06,+.24$] & .007 \\
 & MATH & .17 & .21 & .22 & $+.06$\,[$-.02,+.14$] & .302 \\
 & TQA & .44 & .51 & .50 & $+.06$\,[$-.01,+.12$] & .227 \\
 & ARC & .61 & .66 & .71 & $+.10$\,[$+.02,+.18$] & .022 \\
 & \textbf{pooled} & & & & $\mathbf{+.09}$\,[$+.05,+.13$] & \\
\hline
\end{tabular}%
}
\end{table}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\caption{Control-theoretic audit of each identified $(K,B)$ (Alg.~\ref{alg:audit}). The controllability and observability matrices are full rank ($\mathrm{rank}\,\mathcal C=\mathrm{rank}\,\mathcal O=d=16$) on every cell; both Gramian condition numbers are shown ($\log_{10}$). $\rho(A)<1$ throughout; only Gemma is near-unit-root.}
\label{tab:struct}
\begin{tabular}{llcccc}
\hline
Model & Bench & $\rho(A)$ & $\log_{10}\kappa(W_H)$ & $\log_{10}\kappa(M_H)$ & $\ker M_H$ \\
\hline
Qwen2.5-7B & MMLU & 0.77 & 6.6 & 4.9 & $\{0\}$ \\
 & MATH & 0.75 & 6.4 & 5.3 & $\{0\}$ \\
 & TQA & 0.80 & 7.2 & 5.3 & $\{0\}$ \\
 & ARC & 0.74 & 6.6 & 5.6 & $\{0\}$ \\
\hline
Mistral-7B & MMLU & 0.77 & 6.8 & 4.6 & $\{0\}$ \\
 & MATH & 0.70 & 7.0 & 4.6 & $\{0\}$ \\
 & TQA & 0.74 & 6.6 & 5.3 & $\{0\}$ \\
 & ARC & 0.71 & 7.0 & 6.5 & $\{0\}$ \\
\hline
Llama-3.1-8B & MMLU & 0.66 & 6.4 & 6.2 & $\{0\}$ \\
 & MATH & 0.62 & 6.2 & 5.6 & $\{0\}$ \\
 & TQA & 0.79 & 5.7 & 5.9 & $\{0\}$ \\
 & ARC & 0.70 & 6.9 & 5.4 & $\{0\}$ \\
\hline
Gemma-2-9B & MMLU & 0.99 & 6.4 & 5.7 & $\{0\}$ \\
 & MATH & 0.98 & 6.4 & 4.9 & $\{0\}$ \\
 & TQA & 0.96 & 6.3 & 4.2 & $\{0\}$ \\
 & ARC & 1.00 & 6.4 & 5.5 & $\{0\}$ \\
\hline
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Per-answer reachability from a uniform prior (Thm.~\ref{thm:controllability}(iii)): belief mass installable on each target $x$. ``idx'' is the mean manipulability index and ``min'' the least-reachable answer. Near-unreachable minima ($\le0.1$, in bold) are genuine positional biases (balanced push counts).}
\label{tab:reach}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcccccc}
\hline
Model & Bench & $x_0$ & $x_1$ & $x_2$ & $x_3$ & idx & min \\
\hline
Qwen2.5-7B & MMLU & .75 & .97 & .99 & \textbf{.04} & .69 & \textbf{.04} \\
 & MATH & 1.0 & 1.0 & .99 & .84 & .96 & .84 \\
 & TQA & .99 & .99 & .93 & \textbf{.08} & .75 & \textbf{.08} \\
 & ARC & .99 & 1.0 & .99 & \textbf{.07} & .76 & \textbf{.07} \\
\hline
Mistral-7B & MMLU & 1.0 & .37 & .99 & .20 & .64 & .20 \\
 & MATH & 1.0 & \textbf{.06} & 1.0 & .99 & .76 & \textbf{.06} \\
 & TQA & 1.0 & .97 & .76 & .95 & .92 & .76 \\
 & ARC & 1.0 & .84 & \textbf{.10} & .76 & .67 & \textbf{.10} \\
\hline
Llama-3.1-8B & MMLU & .95 & .90 & .46 & .33 & .66 & .33 \\
 & MATH & .84 & .94 & .25 & \textbf{.10} & .53 & \textbf{.10} \\
 & TQA & .92 & .92 & .44 & .64 & .73 & .44 \\
 & ARC & .97 & .91 & .67 & .25 & .70 & .25 \\
\hline
Gemma-2-9B & MMLU & .73 & .79 & .78 & .19 & .62 & .19 \\
 & MATH & .57 & .98 & .80 & .61 & .74 & .57 \\
 & TQA & .60 & .89 & .77 & .63 & .73 & .60 \\
 & ARC & .75 & .77 & .96 & .47 & .74 & .47 \\
\hline
\end{tabular}%
}
\end{table}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Manipulation and label-free defense. ASR is the single-adversary attack-success rate (no-def), ASR$_{\mathrm d}$ after the honest-margin defense on the same channel; robustness $=1-$ASR$_{\mathrm d}$. $P(\hat c{=}a^\star)$ is how often the defender's chosen push coincides with the truth--low values confirm suppression, not truth-injection (Prop.~\ref{prop:truth}).}
\label{tab:asr}
\begin{tabular}{llccccc}
\hline
Model & Bench & ASR & ASR$_{\mathrm d}$ & $1-$ASR$_{\mathrm d}$ & $\Delta$ASR & $P(\hat c{=}a^\star)$ \\
\hline
Qwen2.5-7B & MMLU & .37 & .01 & .99 & $-.36$ & .31 \\
 & MATH & .48 & .02 & .98 & $-.46$ & .21 \\
 & TQA & .47 & .01 & .99 & $-.46$ & .29 \\
 & ARC & .21 & .01 & .99 & $-.20$ & .38 \\
\hline
Mistral-7B & MMLU & .58 & .09 & .91 & $-.49$ & .26 \\
 & MATH & .58 & .06 & .94 & $-.52$ & .26 \\
 & TQA & .60 & .13 & .87 & $-.47$ & .20 \\
 & ARC & .46 & .07 & .93 & $-.39$ & .34 \\
\hline
Llama-3.1-8B & MMLU & .46 & .06 & .94 & $-.40$ & .30 \\
 & MATH & .51 & .07 & .93 & $-.44$ & .23 \\
 & TQA & .43 & .06 & .94 & $-.38$ & .34 \\
 & ARC & .26 & .07 & .93 & $-.19$ & .44 \\
\hline
Gemma-2-9B & MMLU & .38 & .07 & .93 & $-.31$ & .29 \\
 & MATH & .56 & .13 & .87 & $-.42$ & .18 \\
 & TQA & .30 & .06 & .94 & $-.24$ & .28 \\
 & ARC & .18 & .04 & .96 & $-.13$ & .36 \\
\hline
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\caption{Identification fidelity: held-out control-horizon belief MSE ($t\ge k_{\mathrm{def}}$) for persistence, the autonomous DeGroot/FJ model, the identified $(K,B)$, and a nonlinear MLP. $(K,B)$ improves over DeGroot/FJ on $5/7$ Qwen and $11/12$ replication cells (Koopman winner in bold); the MLP overfits at this sample size.}
\label{tab:fid}
\begin{tabular}{llcccc}
\hline
Model & Bench & persist & DeGroot & Koopman & MLP \\
\hline
Qwen2.5-7B & MMLU & .110 & .144 & \textbf{.121} & .145 \\
 & MATH & .145 & .165 & .175 & .225 \\
 & TQA & .096 & .147 & \textbf{.100} & .127 \\
 & ARC & .110 & .125 & \textbf{.096} & .163 \\
\hline
Mistral-7B & MMLU & .119 & .134 & \textbf{.124} & .133 \\
 & MATH & .137 & .178 & \textbf{.126} & .189 \\
 & TQA & .109 & .141 & \textbf{.132} & .202 \\
 & ARC & .120 & .136 & .140 & .189 \\
\hline
Llama-3.1-8B & MMLU & .066 & .064 & \textbf{.038} & .084 \\
 & MATH & .062 & .076 & \textbf{.059} & .097 \\
 & TQA & .058 & .071 & \textbf{.045} & .104 \\
 & ARC & .063 & .079 & \textbf{.051} & .109 \\
\hline
Gemma-2-9B & MMLU & .068 & .085 & \textbf{.053} & .091 \\
 & MATH & .083 & .097 & \textbf{.084} & .120 \\
 & TQA & .064 & .074 & \textbf{.053} & .095 \\
 & ARC & .079 & .097 & \textbf{.055} & .108 \\
\hline
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\caption{Objective ablation: oracle gain $\Delta$ (oracle$-$no-def) under the older suppress-adversary objective ($-m_{x_{\mathrm{adv}}}$) versus the label-free honest margin ($m_{c_{\mathrm{hon}}}-m_{x_{\mathrm{adv}}}$). The honest margin is positive on all $16$ cells; the suppress-only objective is $\le0$ on $5$ (in bold).}
\label{tab:abl}
\begin{tabular}{llcc}
\hline
Model & Bench & $\Delta$ suppress-adv & $\Delta$ honest-margin \\
\hline
Qwen2.5-7B & MMLU & $\mathbf{-.02}$ & $+.17$ \\
 & MATH & $+.10$ & $+.14$ \\
 & TQA & $+.13$ & $+.12$ \\
 & ARC & $\mathbf{-.02}$ & $+.13$ \\
\hline
Mistral-7B & MMLU & $+.18$ & $+.20$ \\
 & MATH & $+.18$ & $+.28$ \\
 & TQA & $+.14$ & $+.11$ \\
 & ARC & $+.14$ & $+.19$ \\
\hline
Llama-3.1-8B & MMLU & $+.11$ & $+.13$ \\
 & MATH & $+.19$ & $+.19$ \\
 & TQA & $+.08$ & $+.16$ \\
 & ARC & $\mathbf{-.04}$ & $+.06$ \\
\hline
Gemma-2-9B & MMLU & $+.11$ & $+.14$ \\
 & MATH & $+.08$ & $+.06$ \\
 & TQA & $+.13$ & $+.06$ \\
 & ARC & $+.03$ & $+.10$ \\
\hline
\end{tabular}
\end{table}

\section{Discussion and Conclusion}
\label{sec:conclusion}
Reading multi-agent deliberation as a controlled dynamical system turns a set of prompt-level heuristics into a small toolkit organized around one identified object, the pair $(K,B)$: one \emph{attacks} by driving $B$ toward a wrong answer, \emph{audits} by testing controllability and reading off per-answer manipulability, and \emph{defends} by routing the same $B$ to a label-free objective. Attack and moderation are literally one channel with opposite targets (Cor.~\ref{cor:duality}), so a debate that can be moderated can be manipulated to the same degree--a duality the certificate makes quantitative and answer-specific.

The channel is truth-agnostic (Prop.~\ref{prop:truth}): $B$ carries no privileged coordinate for the correct answer, and our defender recovers accuracy by suppressing the adversary rather than by injecting truth. This suggests a mechanism for the empirical observation that simple majority voting often matches or beats elaborate debate \cite{ref17}: if deliberation has no truth-seeking drift and merely propagates whatever the input channel carries, additional rounds add exposure to manipulation rather than signal. The control view sharpens this into a testable statement--debate helps when it suppresses a confident wrong minority and hurts when it amplifies one--both visible here as the spread between the no-defense and oracle columns of Table~\ref{tab:main}.

Several limitations bound the claims. The models are 7--9B and run locally, and per-cell samples are $n\approx90$, which is why we rely on pooled significance and read the borderline Gemma benchmarks as directional. The intervention is a single answer-push; richer channels--evidence, framing, persona--are unmodeled. The autonomous forecast is no better than persistence because beliefs are quasi-static, most sharply on Gemma with $\rho(A)\approx1$, so we validate $(K,B)$ through the audit and the oracle/defense outcomes rather than one-step prediction. The manipulability certificate is computable but only weakly predictive of realized reach, and the dictionary is degree-one; whether a richer dictionary closes the controllability--observability gap, or that gap is fundamental, is unresolved.

To our knowledge this is the first Koopman treatment of an LLM multi-agent system. It yields a compact, falsifiable picture: in belief coordinates deliberation is a stable, controllable, but truth-agnostic linear system whose input channel is identifiable and whose per-answer manipulability is certifiable--yet whose oracle headroom is recovered by no observation-based policy we tried. We leave that gap, $\Delta_{\mathrm{co}}>0$ despite the state being classically observable ($\ker M_H=\{0\}$), as the central open problem: characterizing when what a moderator can \emph{observe} suffices to choose what it can provably \emph{control}.

\appendix
\section{Proof Sketches}
\label{app:proofs}
\emph{Prop.~\ref{prop:consensus}:} $L^2$ absorption decomposition; transient modes vanish geometrically, unimodular non-unit modes in Ces\`aro mean, leaving projection onto $E_1$.
\emph{Prop.~\ref{prop:stability}:} discrete Lyapunov from $\rho(A)<1$ and a Jordan bound on \eqref{eq:mode}.
\emph{Thm.~\ref{thm:controllability}:} discrete controllability theorem; image of the unit ball under $\mathcal C_H$ (Gram matrix $W_H$); rollout of \eqref{eq:edmdc-model}; min-norm solution of $\mathcal C_Hu=\zeta$.
\emph{Prop.~\ref{prop:truth}:} $a^\star$ absent from \eqref{eq:intro-dynamics},\eqref{eq:edmdc}; pooling over uniformly permuted answer positions makes $\mathbb E[\widehat Be_x]$ target-invariant.
\emph{Prop.~\ref{prop:gap}:} the sup in \eqref{eq:co-gap} is attained by $\pi^\star(\psi)=\arg\max_u\mathbb E[m_{a^\star}\mid\Psi(z)=\psi,u]$, equal to the oracle iff the conditioning $\sigma$-algebras induce the same argmax a.s.




%{\appendices
%\section*{Proof of the First Zonklar Equation}
%Appendix one text goes here.
% You can choose not to have a title for an appendix if you want by leaving the argument blank
%\section*{Proof of the Second Zonklar Equation}
%Appendix two text goes here.}



\begin{thebibliography}{1}
\bibliographystyle{IEEEtran}

\bibitem{ref1}
Y. Du, S. Li, A. Torralba, J.~B. Tenenbaum, and I. Mordatch, ``Improving factuality and reasoning in language models through multiagent debate,'' in \textit{Proc. 41st Int. Conf. Machine Learning (ICML)}, 2024.

\bibitem{ref2}
T. Liang, Z. He, W. Jiao, X. Wang, Y. Wang, R. Wang, Y. Yang, and Z. Tu, ``Encouraging divergent thinking in large language models through multi-agent debate,'' in \textit{Proc. 2024 Conf. Empirical Methods in Natural Language Processing (EMNLP)}, 2024.

\bibitem{ref3}
C.-M. Chan, W. Chen, Y. Su, J. Yu, W. Xue, S. Zhang, J. Fu, and Z. Liu, ``ChatEval: Towards better LLM-based evaluators through multi-agent debate,'' \textit{arXiv preprint arXiv:2308.07201}, 2023.

\bibitem{ref4}
A. Smit, P. Duckworth, N. Grinsztajn, T.~D. Barrett, and A. Pretorius, ``Should we be going MAD? A look at multi-agent debate strategies for LLMs,'' in \textit{Proc. 41st Int. Conf. Machine Learning (ICML)}, vol.~235, pp.~45883--45905, 2024.

\bibitem{ref5}
A. Wynn, H. Satija, and G. Hadfield, ``Talk isn't always cheap: Understanding failure modes in multi-agent debate,'' \textit{arXiv preprint arXiv:2509.05396}, 2025.

\bibitem{ref6}
A. Estornell and Y. Liu, ``Multi-LLM debate: Framework, principals, and interventions,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, vol.~37, pp.~28938--28964, 2024.

\bibitem{ref7}
M.~H. DeGroot, ``Reaching a consensus,'' \textit{Journal of the American Statistical Association}, vol.~69, no.~345, pp.~118--121, 1974.


\bibitem{ref8}
N.~E. Friedkin and E.~C. Johnsen, ``Social influence and opinions,'' \textit{Journal of Mathematical Sociology}, vol.~15, no.~3--4, pp.~193--206, 1990.

\bibitem{ref9}
A. Pokharel and R. Dantu, ``Hidden anchors in multi-agent LLM deliberation,'' \textit{arXiv preprint arXiv:2606.19494}, 2026.

\bibitem{ref10}
I. Itkin, ``Delayed verification destabilizes multi-agent LLM belief: Instability thresholds and optimal corrector placement,'' \textit{arXiv preprint arXiv:2606.27409}, 2026.


\bibitem{ref11}
M.~O. Williams, I.~G. Kevrekidis, and C.~W. Rowley, ``A data-driven approximation of the Koopman operator: Extending dynamic mode decomposition,'' \textit{Journal of Nonlinear Science}, vol.~25, no.~6, pp.~1307--1346, 2015.

\bibitem{ref12}
J.~L. Proctor, S.~L. Brunton, and J.~N. Kutz, ``Dynamic mode decomposition with control,'' \textit{SIAM Journal on Applied Dynamical Systems}, vol.~15, no.~1, pp.~142--161, 2016.

\bibitem{ref13}
J.-M. Lasry and P.-L. Lions, ``Jeux \`a champ moyen. I. Le cas stationnaire,'' \textit{Comptes Rendus Math\'ematique}, vol.~343, no.~9, pp.~619--625, 2006.

\bibitem{ref14}
J.-M. Lasry and P.-L. Lions, ``Mean field games,'' \textit{Japanese Journal of Mathematics}, vol.~2, no.~1, pp.~229--260, 2007.

\bibitem{ref15}
C. Bakker, ``Multi-agent, multi-scale systems with the Koopman operator,'' \textit{arXiv preprint arXiv:2506.15236}, 2025.


\bibitem{ref16}
L. Shi, G. Haggerty, and K. Karydis, ``Koopman operators in robot learning,'' \textit{arXiv preprint arXiv:2408.04200}, 2024.

\bibitem{ref17}
H.~K. Choi, X. Zhu, and S. Li, ``Debate or vote: Which yields better decisions in multi-agent large language models?,'' in \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2025.


\bibitem{ref18}
A. Taubenfeld, Y. Dover, R. Reichart, and A. Goldstein, ``Systematic biases in LLM simulations of debates,'' \textit{arXiv preprint arXiv:2402.04049}, 2024.

\bibitem{ref19}
Y.-S. Chuang, R. Tu, C. Dai, Y. Li, S. Vasani, B. Yao, M.~H. Tessler, S. Yang, D. Shah, R. Hawkins, J. Hu, and T.~T. Rogers, ``DEBATE: A large-scale benchmark for evaluating opinion dynamics in role-playing LLM agents,'' \textit{arXiv preprint arXiv:2510.25110}, 2026.


\bibitem{ref20}
I. Mezi\'c and A. Banaszuk, ``Comparison of systems with complex behavior,'' \textit{Physica D: Nonlinear Phenomena}, vol.~197, no.~1--2, pp.~101--133, 2004.

\bibitem{ref21}
S. Klus, P. Koltai, and C. Sch\"utte, ``On the numerical approximation of the Perron--Frobenius and Koopman operator,'' \textit{Journal of Computational Dynamics}, vol.~3, no.~1, pp.~51--79, 2016.

\bibitem{ref22}
E. Kaiser, J.~N. Kutz, and S.~L. Brunton, ``Data-driven discovery of Koopman eigenfunctions for control,'' \textit{Machine Learning: Science and Technology}, vol.~2, no.~3, p.~035023, 2021.


\end{thebibliography}


\newpage

\section{Biography Section}
If you have an EPS/PDF photo (graphicx package needed), extra braces are
 needed around the contents of the optional argument to biography to prevent
 the LaTeX parser from getting confused when it sees the complicated
 $\backslash${\tt{includegraphics}} command within an optional argument. (You can create
 your own custom macro containing the $\backslash${\tt{includegraphics}} command to make things
 simpler here.)
 
\vspace{11pt}

\bf{If you include a photo:}\vspace{-33pt}
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{fig1}}]{Michael Shell}
Use $\backslash${\tt{begin\{IEEEbiography\}}} and then for the 1st argument use $\backslash${\tt{includegraphics}} to declare and link the author photo.
Use the author name as the 3rd argument followed by the biography text.
\end{IEEEbiography}

\vspace{11pt}

\bf{If you will not include a photo:}\vspace{-33pt}
\begin{IEEEbiographynophoto}{John Doe}
Use $\backslash${\tt{begin\{IEEEbiographynophoto\}}} and the author name as the argument followed by the biography text.
\end{IEEEbiographynophoto}




\vfill

\end{document}


