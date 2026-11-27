import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

interface EnginePreset {
  name: string;
  scope?: 'all' | 'part1' | 'part2';
  vMean: number;
  vSd: number;
  naMean: number;
  naSd: number;
  rhoVN: number;
  attrition: number;
  rhoNoShowV: number;
  eukMean: number;
  eukSd: number;
  digMean: number;
  digSd: number;
  essayMean: number;
  essaySd: number;
  rhoEukDig: number;
  rhoEukEssay: number;
  rhoDigEssay: number;
  rhoVP2: number;
}

interface Calibrated3PL {
  effectiveMean: number;
  effectiveSd: number;
  sdMin: number;
  sdMax: number;
  clampedSd: 'min' | 'max' | null;
  a: number;
  bBar: number;
  reliability: number;
}

interface SimulationResult {
  estPart1Passers: number;
  estPart2Sitters: number;
  pEukPass: number;
  estEukPassers: number;
  pDigPass: number;
  estDigPassers: number;
  estPart2McqPassers: number;
  estMarkedEssayPassers: number;
  baselinePasserOdds: number;
  baselinePart2Odds: number;
  part2PoolStats: {
    eukMean: number;
    eukSd: number;
    digMean: number;
    digSd: number;
    essayMean: number;
    essaySd: number;
  };
  part2SubCharts: {
    euk: Array<{ score: number; sitters: number; marked: number }>;
    dig: Array<{ score: number; sitters: number; marked: number }>;
    essay: Array<{ score: number; sitters: number; marked: number }>;
  };
  matrixAdjusted: boolean;
  prelimCutoff: number;
  finalCutoff: number;
  prelimUndersubscribed: boolean;
  finalUndersubscribed: boolean;
  finalTieKeepFrac: number;
  markedEukMean: number;
  markedDigMean: number;
  markedEssayMean: number;
  markedEssayPassRate: number;
  digBench85: number;
  greenThresholdV: number;
  part1: {
    pVerbalPass: number;
    pNAPass: number;
    pJointPass: number;
    obsCorrVN: number;
    verbalAllPmf: Float64Array;
    verbalPassPmf: Float64Array;
  };
  noShowByV: Float64Array;
  obsCorrNoShowV: number;
  part2SitterVMean: number;
  verbalBreakdown: Array<{
    v: number;
    estPassers: number;
    noShowRate: number;
    estPart2Sitters: number;
    pNAGivenV: number;
    condRlOdds: number;
    uncorrRlOdds: number;
    estInReserveList: number;
    shareOfReserveList: number;
    reqEuk: number;
    reqEukPct: number | null;
    prelimGateBinds: boolean;
  }>;
  verbalChartBars: Array<{
    score: number;
    sitters: number;
    passers: number;
    noShowRate: number;
    part2Sitters: number;
    noShows: number;
    pNAGivenV: number;
    condRlOdds: number;
    zone: string;
  }>;
  evaluatePersonalScore: (v: number, euk: number, dig: number, essay: number) => {
    passEuk: boolean;
    passDig: boolean;
    passEssay: boolean;
    prelimScore: number;
    finalScore: number;
    clearsPrelimGate: boolean;
    blockedAtGateWithStrongEssay: boolean;
    verdict: string;
    percentiles: {
      verbalPart1: number;
      eukPart2: number;
      digPart2: number;
      essayPart2: number;
    };
  };
}

interface AD5EngineApi {
  CONSTANTS: {
    REGISTERED: number;
    N_SIT: number;
    N_RL: number;
    N_PRELIM: number;
    V_MEAN_MIN: number;
    NA_MEAN_MIN: number;
  };
  PART1_KEYS: readonly string[];
  PART2_KEYS: readonly string[];
  PRESETS: Record<string, EnginePreset>;
  TEST_SPECS: Record<string, { nItems: number; sigmaB: number; bOffsets: Float64Array; c: Float64Array; cSum: number }>;
  calcPrelimScore: (v: number, euk: number, dig: number) => number;
  calcFinalScore: (v: number, euk: number, dig: number, essay: number) => number;
  normCdf: (z: number) => number;
  calibrate3PL: (key: string, mean: number, sd: number, passMark: number) => Calibrated3PL;
  computePart1Exact: (
    vCal: Calibrated3PL,
    naCal: Calibrated3PL,
    rhoVN: number
  ) => {
    pVerbalPass: number;
    pNAPass: number;
    pJointPass: number;
    obsCorrVN: number;
    verbalAllPmf: Float64Array;
    verbalPassPmf: Float64Array;
    condNAPassGivenV: Float64Array;
  };
  essayMoments: (muStar: number, sdStar: number) => { mean: number; sd: number };
  calibrateEssay: (
    targetMean: number,
    targetSd: number
  ) => {
    muStar: number;
    sdStar: number;
    effectiveMean: number;
    effectiveSd: number;
    clampedSd: 'min' | 'max' | null;
  };
  solveMeansForPassRate: (
    state: EnginePreset,
    targetPassRate: number
  ) => { vMean: number; naMean: number; achievedPassRate: number; clampNote: string };
  ensurePsdAndCholesky: (mat: number[][]) => {
    corr: number[][];
    L: number[][];
    wasAdjusted: boolean;
    minEig: number;
  };
  encodeScenarioParams: (params: EnginePreset) => string;
  buildShareUrl: (baseHref: string, params: EnginePreset) => string;
  decodeScenarioParams: (queryString: string) => {
    params: EnginePreset;
    hasOverrides: boolean;
    activePart1Preset: string | null;
    activePart2Preset: string | null;
  };
  formatScenarioSummary: (params: EnginePreset, sim: SimulationResult, shareUrl?: string) => string;
  runSimulation: (params: EnginePreset) => SimulationResult;
}

function loadEngineFromIndexHtml(): AD5EngineApi {
  const htmlPath = path.resolve(process.cwd(), 'index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const match = html.match(/<script id="irt-engine">([\s\S]*?)<\/script>/);
  assert.ok(match, 'Could not find <script id="irt-engine"> in index.html');
  const sandbox: { window?: Record<string, unknown>; AD5Engine?: AD5EngineApi } = {};
  sandbox.window = sandbox;
  vm.runInNewContext(match[1], sandbox);
  assert.ok(sandbox.AD5Engine, 'AD5Engine was not attached to window');
  return sandbox.AD5Engine;
}

const E = loadEngineFromIndexHtml();

test('Official EPSO AD5 2026 constants and exact scoring formulas', () => {
  assert.equal(E.CONSTANTS.REGISTERED, 174382);
  assert.equal(E.CONSTANTS.N_SIT, 75274);
  assert.equal(E.CONSTANTS.N_RL, 1490);
  assert.equal(E.CONSTANTS.N_PRELIM, 2235);

  // Max scores must equal exactly 100
  assert.equal(E.calcPrelimScore(20, 30, 40), 100);
  assert.ok(Math.abs(E.calcFinalScore(20, 30, 40, 10) - 100) < 1e-12);

  // Midpoint check: V=16, EUK=24, Dig=32, EUFTE=7.0
  assert.equal(E.calcPrelimScore(16, 24, 32), 2 * 16 + 24 + 0.75 * 32);
  assert.ok(
    Math.abs(E.calcFinalScore(16, 24, 32, 7.0) - (1.75 * 16 + (5 / 6) * 24 + 0.625 * 32 + 1.5 * 7.0)) <
      1e-12
  );
});

test('3PL IRT calibration matches target mean/SD and clamps infeasible SDs cleanly', () => {
  const cal = E.calibrate3PL('verbal', 10.5, 4.0, 10);
  assert.ok(Math.abs(cal.effectiveMean - 10.5) < 0.02, `Mean mismatch: ${cal.effectiveMean}`);
  assert.ok(Math.abs(cal.effectiveSd - 4.0) < 0.03, `SD mismatch: ${cal.effectiveSd}`);
  assert.equal(cal.clampedSd, null);
  assert.ok(cal.reliability > 0.5 && cal.reliability < 0.95);

  // Infeasibly low SD must clamp to min floor
  const lowCal = E.calibrate3PL('dig', 30.0, 1.0, 20);
  assert.equal(lowCal.clampedSd, 'min');
  assert.ok(lowCal.effectiveSd > 1.5);
});

test('Baseline simulation produces realistic ~42.5% Part 1 pass rate, attenuated r_obs, exact PSD matrix, and monotonic RL odds', () => {
  assert.equal(E.PRESETS.baseline.attrition, 32);
  assert.equal(E.PRESETS.baseline.rhoNoShowV, -0.36);
  assert.equal(E.PRESETS.baseline.scope, 'all');
  const sim = E.runSimulation(E.PRESETS.baseline);

  // Naturally positive-definite 5x5 correlation matrix
  assert.equal(sim.matrixAdjusted, false);

  // Part 1 pass rate near 42.5%
  assert.ok(
    Math.abs(sim.part1.pJointPass - 0.425) <= 0.006,
    `Unexpected baseline joint pass rate: ${sim.part1.pJointPass}`
  );
  // Observed correlation strictly attenuated below latent rhoVN = 0.65
  assert.ok(
    sim.part1.obsCorrVN > 0.35 && sim.part1.obsCorrVN < 0.54,
    `Unexpected observed correlation: ${sim.part1.obsCorrVN}`
  );
  // Marked Top 2,235 essay mean must be strictly higher than pool mean (5.5) due to positive correlation
  assert.ok(
    sim.markedEssayMean > E.PRESETS.baseline.essayMean + 0.25,
    `Expected selection lift in marked essay mean, got ${sim.markedEssayMean}`
  );
  // Conditional RL odds monotonically increase with Verbal score from 13 to 20
  const rows13to20 = sim.verbalBreakdown.filter((r) => r.v >= 13);
  for (let i = 1; i < rows13to20.length; i++) {
    assert.ok(
      rows13to20[i].condRlOdds >= rows13to20[i - 1].condRlOdds - 0.005,
      `Non-monotonic odds at v=${rows13to20[i].v}`
    );
  }
  // Dynamic green zone threshold must be sensible (between 14 and 18)
  assert.ok(sim.greenThresholdV >= 14 && sim.greenThresholdV <= 18);
  // Deterministic CRN: re-running yields identical cutoffs
  const sim2 = E.runSimulation(E.PRESETS.baseline);
  assert.equal(sim.prelimCutoff, sim2.prelimCutoff);
  assert.equal(sim.finalCutoff, sim2.finalCutoff);
});

test('All Part 1 and Part 2 presets hit calibrated pass-rate regimes, scopes, PSD compliance, and cutoff ordering', () => {
  assert.deepEqual([...E.PART1_KEYS], ['vMean', 'vSd', 'naMean', 'naSd', 'rhoVN']);
  assert.equal(E.PART2_KEYS.length, 12);

  const baseSim = E.runSimulation(E.PRESETS.baseline);
  const vEasierSim = E.runSimulation(E.PRESETS.jppVerbalEasier);
  const naEasierSim = E.runSimulation(E.PRESETS.jppNaEasier);
  const lowPassSim = E.runSimulation(E.PRESETS.lowPassRate);
  const highPassSim = E.runSimulation(E.PRESETS.highPassRate);
  const p2BaseSim = E.runSimulation(E.PRESETS.part2Baseline);
  const ceilingSim = E.runSimulation(E.PRESETS.ceilingCrunch);
  const harshSim = E.runSimulation(E.PRESETS.harshFatigue);

  // Part 1 symmetry & baseline defaults
  assert.equal(E.PRESETS.baseline.vMean, 10.5);
  assert.equal(E.PRESETS.baseline.naMean, 10.5);
  assert.equal(E.PRESETS.jppNaEasier.name, 'JPP rate & NA easier');
  for (const p of [E.PRESETS.baseline, E.PRESETS.jppVerbalEasier, E.PRESETS.jppNaEasier, E.PRESETS.lowPassRate, E.PRESETS.highPassRate]) {
    assert.equal(p.vSd, p.naSd, `Expected equal Verbal and Num+Abs SD in ${p.name}`);
  }

  // Scopes & natural PSD
  assert.equal(E.PRESETS.jppVerbalEasier.scope, 'part1');
  assert.equal(E.PRESETS.jppNaEasier.scope, 'part1');
  assert.equal(E.PRESETS.lowPassRate.scope, 'part1');
  assert.equal(E.PRESETS.highPassRate.scope, 'part1');
  assert.equal(E.PRESETS.part2Baseline.scope, 'part2');
  assert.equal(E.PRESETS.ceilingCrunch.scope, 'part2');
  assert.equal(E.PRESETS.harshFatigue.scope, 'part2');

  for (const s of [baseSim, vEasierSim, naEasierSim, lowPassSim, highPassSim, p2BaseSim, ceilingSim, harshSim]) {
    assert.equal(s.matrixAdjusted, false);
  }

  // Joint pass rates within 42.5% +/- 1.0% for the three JPP presets
  for (const s of [baseSim, vEasierSim, naEasierSim]) {
    assert.ok(
      Math.abs(s.part1.pJointPass - 0.425) <= 0.01,
      `Expected ~42.5% joint pass, got ${(s.part1.pJointPass * 100).toFixed(2)}%`
    );
  }

  // Asymmetric marginal pass rates (> 12 percentage points separation)
  assert.ok(
    vEasierSim.part1.pVerbalPass > vEasierSim.part1.pNAPass + 0.12,
    `jppVerbalEasier marginal separation too small: V=${vEasierSim.part1.pVerbalPass}, NA=${vEasierSim.part1.pNAPass}`
  );
  assert.ok(
    naEasierSim.part1.pNAPass > naEasierSim.part1.pVerbalPass + 0.12,
    `jppNaEasier marginal separation too small: V=${naEasierSim.part1.pVerbalPass}, NA=${naEasierSim.part1.pNAPass}`
  );

  // Low pass rate within 33.0% +/- 1.0% and High pass rate within 60.0% +/- 1.0%
  assert.ok(
    Math.abs(lowPassSim.part1.pJointPass - 0.33) <= 0.01,
    `Expected ~33.0% low pass rate, got ${(lowPassSim.part1.pJointPass * 100).toFixed(2)}%`
  );
  assert.ok(
    Math.abs(highPassSim.part1.pJointPass - 0.60) <= 0.01,
    `Expected ~60.0% high pass rate, got ${(highPassSim.part1.pJointPass * 100).toFixed(2)}%`
  );

  // Part 2 baseline matches full baseline cutoffs
  assert.equal(p2BaseSim.finalCutoff, baseSim.finalCutoff);
  assert.equal(p2BaseSim.prelimCutoff, baseSim.prelimCutoff);

  // Strict Part 2 final cutoff ordering: harshFatigue < baseline < ceilingCrunch
  assert.ok(
    harshSim.finalCutoff < baseSim.finalCutoff && baseSim.finalCutoff < ceilingSim.finalCutoff,
    `Expected harsh (${harshSim.finalCutoff}) < baseline (${baseSim.finalCutoff}) < ceiling (${ceilingSim.finalCutoff})`
  );
});

test('Bidirectional Joint Pass Rate solver shifts both means equally and clamps gracefully at extremes', () => {
  const base = E.PRESETS.baseline;
  const solved35 = E.solveMeansForPassRate(base, 0.35);
  assert.ok(Math.abs(solved35.achievedPassRate - 0.35) < 0.005);
  // Equal shift preserves (vMean - naMean) = 0.00
  assert.ok(Math.abs(solved35.vMean - solved35.naMean - (base.vMean - base.naMean)) < 0.03);
  assert.equal(solved35.clampNote, '');

  // Extreme pass rate hits bounds cleanly with a clamp note
  const solved99 = E.solveMeansForPassRate(base, 0.99);
  assert.ok(solved99.clampNote.length > 0);
  assert.ok(solved99.vMean <= 16.0 && solved99.naMean <= 16.0);
});

test('PSD matrix projection fixes non-positive-definite correlation matrices', () => {
  // Impossible triangle: A-B=0.9, A-C=0.9, B-C=-0.9 is non-PSD
  const bad = [
    [1, 0.9, 0.9],
    [0.9, 1, -0.9],
    [0.9, -0.9, 1]
  ];
  const res = E.ensurePsdAndCholesky(bad);
  assert.equal(res.wasAdjusted, true);
  // Verify L * L^T reconstructs unit diagonal
  for (let i = 0; i < 3; i++) {
    let diag = 0;
    for (let k = 0; k <= i; k++) diag += res.L[i][k] * res.L[i][k];
    assert.ok(Math.abs(diag - 1.0) < 1e-6);
  }
});

test('Personal score evaluator detects strong essay blocked at the Top 2,235 gate', () => {
  const sim = E.runSimulation(E.PRESETS.baseline);
  // Construct a score just below prelimCutoff (so blocked at Top 2,235) with a 10.0/10 essay that pushes Final above finalCutoff
  const v = 14;
  const dig = 32;
  // Pick EUK so prelim = 2*14 + euk + 24 = 52 + euk is ~1-2 pts below prelimCutoff
  const euk = Math.max(15, Math.min(30, Math.floor(sim.prelimCutoff - 52 - 1.2)));
  const ev = sim.evaluatePersonalScore(v, euk, dig, 10.0);
  assert.equal(ev.passEuk, true);
  assert.equal(ev.passDig, true);
  assert.equal(ev.clearsPrelimGate, false);
  if (ev.finalScore >= sim.finalCutoff) {
    assert.equal(ev.blockedAtGateWithStrongEssay, true);
  }
});

test('Num+Abs uses two independent symmetric 10-item normal quantile grids with unconfounded c', () => {
  const na = E.TEST_SPECS.na;
  assert.equal(na.nItems, 20);
  let sumNumB = 0;
  let sumAbsB = 0;
  for (let i = 0; i < 10; i++) {
    assert.equal(na.c[i], 0.18);
    assert.equal(na.c[i + 10], 0.16);
    assert.ok(Math.abs(na.bOffsets[i] - na.bOffsets[i + 10]) < 1e-12, `Subtest grids differ at item ${i}`);
    sumNumB += na.bOffsets[i];
    sumAbsB += na.bOffsets[i + 10];
  }
  assert.ok(Math.abs(sumNumB) < 1e-10, `Numerical bOffsets not centered: ${sumNumB}`);
  assert.ok(Math.abs(sumAbsB) < 1e-10, `Abstract bOffsets not centered: ${sumAbsB}`);
});

test('Exact normal-CDF essayMoments and calibrateEssay match target moments smoothly', () => {
  assert.ok(Math.abs(E.normCdf(0) - 0.5) < 1e-7);
  assert.ok(Math.abs(E.normCdf(1.95996398454) - 0.975) < 1e-5);
  assert.ok(Math.abs(E.normCdf(-1.95996398454) - 0.025) < 1e-5);

  // Exact symmetry at muStar = 5.0
  const sym = E.essayMoments(5.0, 1.5);
  assert.ok(Math.abs(sym.mean - 5.0) < 1e-6, `Expected exact mean 5.0, got ${sym.mean}`);

  const cal = E.calibrateEssay(5.5, 1.5);
  const mom = E.essayMoments(cal.muStar, cal.sdStar);
  assert.ok(Math.abs(mom.mean - 5.5) < 0.005, `Essay mean mismatch: ${mom.mean}`);
  assert.ok(Math.abs(mom.sd - 1.5) < 0.005, `Essay SD mismatch: ${mom.sd}`);
});

test('Part 2 survivor standardization makes simulated Part 2 sitter pool match sliders and uses Part 2 sitter baseline for green zone', () => {
  const base = E.PRESETS.baseline;
  const sim = E.runSimulation(base);

  // Simulated Part 2 sitter pool means & SDs must closely match user Part 2 sliders
  assert.ok(
    Math.abs(sim.part2PoolStats.eukMean - base.eukMean) < 0.12,
    `EUK pool mean ${sim.part2PoolStats.eukMean} drifted from slider ${base.eukMean}`
  );
  assert.ok(
    Math.abs(sim.part2PoolStats.eukSd - base.eukSd) < 0.15,
    `EUK pool SD ${sim.part2PoolStats.eukSd} drifted from slider ${base.eukSd}`
  );
  assert.ok(
    Math.abs(sim.part2PoolStats.digMean - base.digMean) < 0.15,
    `Dig pool mean ${sim.part2PoolStats.digMean} drifted from slider ${base.digMean}`
  );
  assert.ok(
    Math.abs(sim.part2PoolStats.digSd - base.digSd) < 0.15,
    `Dig pool SD ${sim.part2PoolStats.digSd} drifted from slider ${base.digSd}`
  );
  assert.ok(
    Math.abs(sim.part2PoolStats.essayMean - base.essayMean) < 0.08,
    `Essay pool mean ${sim.part2PoolStats.essayMean} drifted from slider ${base.essayMean}`
  );
  assert.ok(
    Math.abs(sim.part2PoolStats.essaySd - base.essaySd) < 0.08,
    `Essay pool SD ${sim.part2PoolStats.essaySd} drifted from slider ${base.essaySd}`
  );

  // Green zone threshold must compare condRlOdds against baselinePart2Odds (not baselinePasserOdds)
  const expectedP2Odds = Math.min(1.0, E.CONSTANTS.N_RL / sim.estPart2Sitters);
  assert.ok(Math.abs(sim.baselinePart2Odds - expectedP2Odds) < 1e-12);
  assert.ok(sim.baselinePart2Odds > sim.baselinePasserOdds);

  const expectedGreenV = sim.verbalBreakdown.find((r) => r.condRlOdds > sim.baselinePart2Odds)?.v ?? 21;
  assert.equal(sim.greenThresholdV, expectedGreenV);
});

test('High-SD Part 1 quadrature eliminates aliasing at vMean=11.0, vSd=5.0 and solver matches computePart1Exact at SD=5.0', () => {
  // Verify updated slider bounds, Part 2 defaults, scenarios, and Digital Skills guessing floor
  assert.equal(E.CONSTANTS.V_MEAN_MIN, 6.5);
  assert.equal(E.CONSTANTS.NA_MEAN_MIN, 6.5);
  assert.equal(E.PRESETS.baseline.eukMean, 17.5);
  assert.equal(E.PRESETS.baseline.eukSd, 5.0);
  assert.equal(E.PRESETS.baseline.digMean, 25.8);
  assert.equal(E.PRESETS.baseline.digSd, 6.5);
  assert.equal(E.PRESETS.ceilingCrunch.eukMean, 19.3);
  assert.equal(E.PRESETS.ceilingCrunch.digMean, 28.2);
  assert.equal(E.PRESETS.ceilingCrunch.digSd, 6.5);
  assert.equal(E.PRESETS.harshFatigue.name, 'Harsh Calibration');
  assert.equal(E.PRESETS.harshFatigue.attrition, 41);
  assert.equal(E.PRESETS.harshFatigue.eukMean, 14.3);
  assert.equal(E.PRESETS.harshFatigue.digMean, 23.9);
  assert.equal(E.TEST_SPECS.dig.c[0], 0.21);

  // High-SD Verbal distribution must be smooth without artificial dip at 12-13 between peaks at 10 and 15
  const vCal = E.calibrate3PL('verbal', 11.0, 5.0, 10);
  const naCal = E.calibrate3PL('na', 10.6, 5.0, 10);
  const p1 = E.computePart1Exact(vCal, naCal, 0.65);
  const pmf = p1.verbalAllPmf;

  // Check there is no aliasing trough at 12-13 below both 10 and 15
  const minMid = Math.min(pmf[12], pmf[13]);
  const minFlank = Math.min(pmf[10], pmf[15]);
  assert.ok(
    minMid >= minFlank,
    `Expected no artificial dip at 12-13 (${pmf[12].toFixed(4)}, ${pmf[13].toFixed(4)}) below peaks at 10 (${pmf[10].toFixed(4)}) and 15 (${pmf[15].toFixed(4)})`
  );
  // Also verify smooth unimodality/concavity across 10..15 (no local valley at 11..14)
  for (let s = 11; s <= 14; s++) {
    assert.ok(
      !(pmf[s] < pmf[s - 1] && pmf[s] < pmf[s + 1]),
      `Artificial local dip detected at score ${s}: [${pmf[s - 1].toFixed(5)}, ${pmf[s].toFixed(5)}, ${pmf[s + 1].toFixed(5)}]`
    );
  }

  // Verify solveMeansForPassRate agrees with computePart1Exact.pJointPass within 0.003 at vSd=5.0, naSd=5.0
  const highSdState: EnginePreset = {
    ...E.PRESETS.baseline,
    vMean: 11.0,
    vSd: 5.0,
    naMean: 10.6,
    naSd: 5.0
  };
  for (const target of [0.35, 0.45, 0.55]) {
    const solved = E.solveMeansForPassRate(highSdState, target);
    const vSolved = E.calibrate3PL('verbal', solved.vMean, highSdState.vSd, 10);
    const naSolved = E.calibrate3PL('na', solved.naMean, highSdState.naSd, 10);
    const exact = E.computePart1Exact(vSolved, naSolved, highSdState.rhoVN);
    assert.ok(
      Math.abs(exact.pJointPass - target) < 0.003,
      `Solver mismatch at SD=5.0 for target=${target}: exact=${exact.pJointPass.toFixed(5)}, solved=${solved.achievedPassRate.toFixed(5)}`
    );
    assert.ok(
      Math.abs(exact.pJointPass - solved.achievedPassRate) < 0.003,
      `evalPairFast vs computePart1Exact mismatch at SD=5.0: exact=${exact.pJointPass.toFixed(5)}, fast=${solved.achievedPassRate.toFixed(5)}`
    );
  }
});

test('Part 2 subtest distributions and Personal Score Tester percentiles are exact and consistent', () => {
  const sim = E.runSimulation(E.PRESETS.baseline);
  const { euk, dig, essay } = sim.part2SubCharts;

  assert.equal(euk.length, 31);
  assert.equal(dig.length, 41);
  assert.equal(essay.length, 21);

  for (const [name, arr] of [
    ['euk', euk],
    ['dig', dig],
    ['essay', essay]
  ] as const) {
    let totalSitters = 0;
    let totalMarked = 0;
    for (const b of arr) {
      assert.ok(b.marked <= b.sitters + 1e-6, `${name} score ${b.score}: marked (${b.marked}) > sitters (${b.sitters})`);
      totalSitters += b.sitters;
      totalMarked += b.marked;
    }
    assert.ok(
      Math.abs(totalSitters - sim.estPart2Sitters) < 1.0,
      `${name} sitter sum ${totalSitters} != estPart2Sitters ${sim.estPart2Sitters}`
    );
    assert.ok(
      Math.abs(totalMarked - E.CONSTANTS.N_PRELIM) < 5.0,
      `${name} marked sum ${totalMarked} != N_PRELIM (${E.CONSTANTS.N_PRELIM})`
    );
  }

  // Below MCQ pass marks, zero candidates can reach the Top 2,235 essay-marking gate
  for (let s = 0; s < 15; s++) assert.equal(euk[s].marked, 0);
  for (let s = 0; s < 20; s++) assert.equal(dig[s].marked, 0);

  // Selection lift: marked EUK & Digital averages exceed Part 2 sitter averages
  assert.ok(sim.markedEukMean > sim.part2PoolStats.eukMean + 2.0);
  assert.ok(sim.markedDigMean > sim.part2PoolStats.digMean + 2.0);
  assert.ok(sim.markedEssayPassRate > 0.5 && sim.markedEssayPassRate <= 1.0);

  // Per-component EUK & Digital Skills pass rates and passer counts match subtest distributions
  const eukPassSum = euk.filter((b) => b.score >= 15).reduce((acc, b) => acc + b.sitters, 0);
  const digPassSum = dig.filter((b) => b.score >= 20).reduce((acc, b) => acc + b.sitters, 0);
  assert.ok(Math.abs(sim.estEukPassers - eukPassSum) < 1e-3);
  assert.ok(Math.abs(sim.estDigPassers - digPassSum) < 1e-3);
  assert.ok(Math.abs(sim.pEukPass - sim.estEukPassers / sim.estPart2Sitters) < 1e-9);
  assert.ok(Math.abs(sim.pDigPass - sim.estDigPassers / sim.estPart2Sitters) < 1e-9);
  assert.ok(sim.estPart2McqPassers <= Math.min(sim.estEukPassers, sim.estDigPassers) + 1e-6);

  // Personal Score Tester percentiles: Verbal vs Part 1 sitters, EUK/Dig/Essay vs Part 2 sitters
  const lowEv = sim.evaluatePersonalScore(10, 0, 0, 0);
  const midEv = sim.evaluatePersonalScore(16, 23, 30, 6.5);
  const highEv = sim.evaluatePersonalScore(20, 30, 40, 10);

  assert.equal(lowEv.percentiles.eukPart2, 0);
  assert.equal(lowEv.percentiles.digPart2, 0);
  assert.equal(lowEv.percentiles.essayPart2, 0);

  // Exact match between verbalPart1 percentile and cumulative Part 1 PMF strictly below v=16
  let expectedVerbal16 = 0;
  for (let s = 0; s < 16; s++) expectedVerbal16 += sim.part1.verbalAllPmf[s];
  assert.ok(Math.abs(midEv.percentiles.verbalPart1 - expectedVerbal16) < 1e-12);

  // Strict monotonicity across low < mid < high
  assert.ok(lowEv.percentiles.verbalPart1 < midEv.percentiles.verbalPart1 && midEv.percentiles.verbalPart1 < highEv.percentiles.verbalPart1);
  assert.ok(lowEv.percentiles.eukPart2 < midEv.percentiles.eukPart2 && midEv.percentiles.eukPart2 < highEv.percentiles.eukPart2);
  assert.ok(lowEv.percentiles.digPart2 < midEv.percentiles.digPart2 && midEv.percentiles.digPart2 < highEv.percentiles.digPart2);
  assert.ok(lowEv.percentiles.essayPart2 < midEv.percentiles.essayPart2 && midEv.percentiles.essayPart2 < highEv.percentiles.essayPart2);

  // Est. in Reserve List per Verbal score matches Part 2 sitters * condRlOdds and sums near 1,490
  let totalEstRl = 0;
  for (const row of sim.verbalBreakdown) {
    const expected = row.estPassers * (1 - row.noShowRate) * row.condRlOdds;
    assert.ok(Math.abs(row.estInReserveList - expected) < 1e-9);
    assert.ok(Math.abs(row.estPart2Sitters - row.estPassers * (1 - row.noShowRate)) < 1e-9);
    assert.ok(Math.abs(row.shareOfReserveList - row.estInReserveList / E.CONSTANTS.N_RL) < 1e-12);
    totalEstRl += row.estInReserveList;
  }
  assert.ok(
    Math.abs(totalEstRl - E.CONSTANTS.N_RL) < 75,
    `Sum of estInReserveList (${totalEstRl.toFixed(1)}) drifted too far from N_RL (${E.CONSTANTS.N_RL})`
  );
});

test('Verbal-correlated Part 2 no-show hits 32% overall with rho=-0.36 by default, and reduces cutoff softening vs uniform attrition', () => {
  const baseSim = E.runSimulation(E.PRESETS.baseline);
  const uniformSim = E.runSimulation({ ...E.PRESETS.baseline, rhoNoShowV: 0 });

  // Baseline: 32% overall no-show with rho=-0.36 yields ~4.8% no-show at Verbal 20/20
  assert.ok(
    Math.abs(baseSim.noShowByV[20] - 0.048) < 0.005,
    `Expected ~4.8% no-show at V=20, got ${(baseSim.noShowByV[20] * 100).toFixed(2)}%`
  );

  // Passer-weighted average no-show must equal 32.0% exactly
  let weightedNoShow = 0;
  let totalPassers = 0;
  for (const row of baseSim.verbalBreakdown) {
    weightedNoShow += row.estPassers * row.noShowRate;
    totalPassers += row.estPassers;
  }
  assert.ok(Math.abs(weightedNoShow / totalPassers - 0.32) < 1e-9);

  // Strictly decreasing no-show from V=10 to V=20 when rhoNoShowV < 0
  for (let v = 11; v <= 20; v++) {
    assert.ok(
      baseSim.noShowByV[v] < baseSim.noShowByV[v - 1],
      `Expected noShowByV[${v}] (${baseSim.noShowByV[v]}) < noShowByV[${v - 1}] (${baseSim.noShowByV[v - 1]})`
    );
  }

  // Uniform limit when rhoNoShowV === 0
  for (let v = 10; v <= 20; v++) {
    assert.ok(Math.abs(uniformSim.noShowByV[v] - 0.32) < 1e-12);
  }

  // Selective low-Verbal dropouts lift active Part 2 sitter Verbal mean and keep Part 2 cutoffs higher than uniform 32% attrition
  assert.ok(baseSim.part2SitterVMean > uniformSim.part2SitterVMean + 0.4);
  assert.ok(
    baseSim.prelimCutoff > uniformSim.prelimCutoff + 0.4,
    `Expected selective no-show prelimCutoff (${baseSim.prelimCutoff}) > uniform (${uniformSim.prelimCutoff})`
  );
  assert.ok(
    baseSim.finalCutoff > uniformSim.finalCutoff + 0.3,
    `Expected selective no-show finalCutoff (${baseSim.finalCutoff}) > uniform (${uniformSim.finalCutoff})`
  );

  // Part 1 chart bars carry consistent part2Sitters + noShows === passers
  for (const bar of baseSim.verbalChartBars) {
    if (bar.score >= 10) {
      assert.ok(Math.abs(bar.part2Sitters + bar.noShows - bar.passers) < 1e-9);
      assert.ok(Math.abs(bar.noShowRate - baseSim.noShowByV[bar.score]) < 1e-12);
    } else {
      assert.equal(bar.noShowRate, 0);
      assert.equal(bar.part2Sitters, 0);
      assert.equal(bar.noShows, 0);
    }
  }
});

test('Uncorrelated Sitters RL Odds covers Verbal 10..20, is monotonic, and converges to condRlOdds when rhoVP2=0', () => {
  const baseSim = E.runSimulation(E.PRESETS.baseline);
  const zeroCorrSim = E.runSimulation({ ...E.PRESETS.baseline, rhoVP2: 0 });

  assert.equal(baseSim.verbalBreakdown.length, 11);
  assert.equal(baseSim.verbalBreakdown[0].v, 10);
  assert.equal(baseSim.verbalBreakdown[10].v, 20);

  // Verify reqEukPct matches Part 2 sitter EUK percentile (null when > 30/30)
  for (const row of baseSim.verbalBreakdown) {
    if (row.reqEuk > 30) {
      assert.equal(row.reqEukPct, null);
    } else {
      assert.ok(row.reqEukPct !== null && row.reqEukPct >= 0 && row.reqEukPct <= 1);
      const ev = baseSim.evaluatePersonalScore(row.v, row.reqEuk, baseSim.digBench85, 7.0);
      assert.ok(Math.abs(row.reqEukPct - ev.percentiles.eukPart2) < 1e-12);
    }
  }

  // Strictly non-decreasing across Verbal 10..20
  for (let i = 1; i < baseSim.verbalBreakdown.length; i++) {
    assert.ok(
      baseSim.verbalBreakdown[i].uncorrRlOdds >= baseSim.verbalBreakdown[i - 1].uncorrRlOdds,
      `uncorrRlOdds decreased at v=${baseSim.verbalBreakdown[i].v}`
    );
  }

  // With positive rhoVP2=0.45: uncorrelated odds exceed correlated odds at low V (10..12) and trail at high V (16..20)
  for (const row of baseSim.verbalBreakdown) {
    if (row.v <= 12) {
      assert.ok(
        row.uncorrRlOdds >= row.condRlOdds,
        `Expected uncorrRlOdds (${row.uncorrRlOdds}) >= condRlOdds (${row.condRlOdds}) at v=${row.v}`
      );
    }
    if (row.v >= 16) {
      assert.ok(
        row.uncorrRlOdds < row.condRlOdds,
        `Expected uncorrRlOdds (${row.uncorrRlOdds}) < condRlOdds (${row.condRlOdds}) at v=${row.v}`
      );
    }
  }

  // When rhoVP2 === 0, uncorrRlOdds matches condRlOdds within Monte Carlo tolerance
  for (const row of zeroCorrSim.verbalBreakdown) {
    assert.ok(
      Math.abs(row.uncorrRlOdds - row.condRlOdds) < 0.02,
      `At rhoVP2=0, uncorrRlOdds (${row.uncorrRlOdds}) diverged from condRlOdds (${row.condRlOdds}) at v=${row.v}`
    );
  }
});

test('Compact scenario URL encoding, safe decoding/clamping, and plain-text summary formatting work end-to-end', () => {
  // Baseline produces an empty compact query string
  assert.equal(E.encodeScenarioParams(E.PRESETS.baseline), '');

  // Custom scenario round-trips cleanly and stays compact (only non-default keys)
  const custom: EnginePreset = {
    ...E.PRESETS.baseline,
    vMean: 11.8,
    naMean: 11.8,
    attrition: 40,
    rhoNoShowV: -0.5,
    eukMean: 19.3,
    digMean: 28.2,
    essayMean: 6.0,
    rhoVP2: 0.35
  };
  const qs = E.encodeScenarioParams(custom);
  const qsKeys = qs.split('&').map((p) => p.split('=')[0]);
  assert.ok(qs.includes('vm=11.8') && qs.includes('at=40') && qs.includes('rns=-0.5') && qs.includes('rvp=0.35'));
  assert.ok(!qsKeys.includes('vs'), 'Default vSd should be omitted from compact URL');
  assert.ok(!qsKeys.includes('ns'), 'Default naSd should be omitted from compact URL');

  const decoded = E.decodeScenarioParams(`?${qs}`);
  assert.equal(decoded.hasOverrides, true);
  assert.equal(decoded.params.vMean, 11.8);
  assert.equal(decoded.params.naMean, 11.8);
  assert.equal(decoded.params.attrition, 40);
  assert.equal(decoded.params.rhoNoShowV, -0.5);
  assert.equal(decoded.params.eukMean, 19.3);
  assert.equal(decoded.params.digMean, 28.2);
  assert.equal(decoded.params.essayMean, 6.0);
  assert.equal(decoded.params.rhoVP2, 0.35);
  assert.equal(decoded.activePart1Preset, 'highPassRate');
  assert.equal(decoded.activePart2Preset, null);

  // Out-of-range and invalid parameters are safely clamped or ignored
  const unsafe = E.decodeScenarioParams('?vm=999&at=-20&rns=0.5&rvp=notanumber&unknown=123');
  assert.equal(unsafe.hasOverrides, true);
  assert.equal(unsafe.params.vMean, 16.0);
  assert.equal(unsafe.params.attrition, 0);
  assert.equal(unsafe.params.rhoNoShowV, 0);
  assert.equal(unsafe.params.rhoVP2, E.PRESETS.baseline.rhoVP2);

  // Plain-text summary includes means, SDs, correlations, cutoffs, and link (no NaN)
  const sim = E.runSimulation(custom);
  const ghUrl = E.buildShareUrl('https://marcotamb.github.io/AD5_simulation', custom);
  assert.equal(ghUrl, `https://marcotamb.github.io/AD5_simulation/?${qs}`);
  assert.equal(
    E.buildShareUrl('https://marcotamb.github.io/AD5_simulation/index.html#part2', custom),
    `https://marcotamb.github.io/AD5_simulation/?${qs}`
  );
  assert.equal(
    E.buildShareUrl('https://ais-dev-example.europe-west2.run.app/', custom),
    `https://marcotamb.github.io/AD5_simulation/?${qs}`
  );
  assert.equal(
    E.buildShareUrl('http://localhost:3000/', custom),
    `https://marcotamb.github.io/AD5_simulation/?${qs}`
  );
  assert.equal(
    E.buildShareUrl('file:///Users/me/AD5_simulation/index.html', custom),
    `https://marcotamb.github.io/AD5_simulation/?${qs}`
  );
  assert.equal(
    E.buildShareUrl('https://otheruser.github.io/AD5_simulation', custom),
    `https://otheruser.github.io/AD5_simulation/?${qs}`
  );

  // Hash-only and full GitHub Pages URLs decode cleanly, while plain anchors are ignored
  const fromHash = E.decodeScenarioParams('https://example.github.io/ad5-sim/#vm=11.8&at=40');
  assert.equal(fromHash.hasOverrides, true);
  assert.equal(fromHash.params.vMean, 11.8);
  assert.equal(fromHash.params.attrition, 40);
  const fromHashQ = E.decodeScenarioParams('#?vm=11.8&rns=-0.5');
  assert.equal(fromHashQ.hasOverrides, true);
  assert.equal(fromHashQ.params.rhoNoShowV, -0.5);
  const plainAnchor = E.decodeScenarioParams('#part2');
  assert.equal(plainAnchor.hasOverrides, false);

  const summary = E.formatScenarioSummary(custom, sim, ghUrl);
  assert.ok(!summary.includes('NaN'), `Unexpected NaN in summary:\n${summary}`);
  assert.ok(summary.includes('Part 1 (Reasoning):'));
  assert.ok(summary.includes('Part 2 (Ranking):'));
  assert.ok(summary.includes('Part 2 Correlations:'));
  assert.ok(summary.includes('ρ(V,NA)=0.65'));
  assert.ok(summary.includes('ρ(NS,V)=-0.50'));
  assert.ok(summary.includes('ρ(V,P2)=0.35'));
  assert.ok(summary.includes(`Link: ${ghUrl}`));
});

test('solveBBarForMean handles extreme low-a brackets without false clamping and reports accurate sdMax', () => {
  // Digital Skills at extreme high mean (36.0) and low SD (2.5)
  const digExt = E.calibrate3PL('dig', 36.0, 2.5, 20);
  assert.ok(
    Math.abs(digExt.effectiveMean - 36.0) < 0.02,
    `Dig mean drifted at (36.0, 2.5): got ${digExt.effectiveMean}`
  );
  assert.ok(
    Math.abs(digExt.effectiveSd - 2.5) < 0.05,
    `Dig SD drifted or falsely clamped at (36.0, 2.5): got ${digExt.effectiveSd}, clamped=${digExt.clampedSd}`
  );
  assert.equal(digExt.clampedSd, null);
  assert.ok(digExt.sdMax > digExt.sdMin + 1.0);

  // Verbal at extreme low mean (6.5) and low SD (2.1)
  const vExt = E.calibrate3PL('verbal', 6.5, 2.1, 10);
  assert.ok(
    Math.abs(vExt.effectiveMean - 6.5) < 0.02,
    `Verbal mean drifted at (6.5, 2.1): got ${vExt.effectiveMean}`
  );
  assert.ok(
    Math.abs(vExt.effectiveSd - 2.1) < 0.05,
    `Verbal SD drifted or falsely clamped at (6.5, 2.1): got ${vExt.effectiveSd}, clamped=${vExt.clampedSd}`
  );
  assert.equal(vExt.clampedSd, null);

  // When SD is intentionally below sdMin, sdMax must still report the true upper bound (not sdMin)
  const vClampedMin = E.calibrate3PL('verbal', 10.5, 1.0, 10);
  assert.equal(vClampedMin.clampedSd, 'min');
  assert.ok(
    vClampedMin.sdMax > vClampedMin.sdMin + 2.0,
    `Expected sdMax (${vClampedMin.sdMax}) > sdMin (${vClampedMin.sdMin}) + 2.0`
  );
});

test('Undersubscribed competition cutoffs floor cleanly at 50.00 and never eliminate passing candidates', () => {
  // Low essay mean -> fewer than 1,490 marked essay passers
  const lowEssaySim = E.runSimulation({ ...E.PRESETS.baseline, essayMean: 4.0, essaySd: 0.8 });
  assert.equal(lowEssaySim.finalUndersubscribed, true);
  assert.equal(lowEssaySim.finalCutoff, 50.0);
  assert.equal(lowEssaySim.finalTieKeepFrac, 1.0);
  assert.ok(lowEssaySim.estMarkedEssayPassers < E.CONSTANTS.N_RL);

  // Any candidate who clears prelim gate and passes EUFTE >= 5.0 must pass Final cutoff
  const strongMcqMinEssay = lowEssaySim.evaluatePersonalScore(18, 28, 36, 5.0);
  assert.equal(strongMcqMinEssay.clearsPrelimGate, true);
  assert.equal(strongMcqMinEssay.passEssay, true);
  assert.equal(strongMcqMinEssay.verdict, 'pass');

  // Extreme low Part 2 MCQ means -> fewer than 2,235 MCQ passers
  const lowMcqSim = E.runSimulation({
    ...E.PRESETS.baseline,
    attrition: 60,
    eukMean: 10.0,
    eukSd: 2.5,
    digMean: 15.0,
    digSd: 2.5
  });
  assert.equal(lowMcqSim.prelimUndersubscribed, true);
  assert.equal(lowMcqSim.prelimCutoff, 50.0);
  assert.equal(lowMcqSim.finalUndersubscribed, true);
  assert.equal(lowMcqSim.finalCutoff, 50.0);
  // Minimum passing scores across all tests (10, 15, 20, 5.0 -> exact 50.00 / 50.00) must pass
  const minPasser = lowMcqSim.evaluatePersonalScore(10, 15, 20, 5.0);
  assert.equal(minPasser.prelimScore, 50.0);
  assert.equal(minPasser.finalScore, 50.0);
  assert.equal(minPasser.clearsPrelimGate, true);
  assert.equal(minPasser.verdict, 'pass');
});

test('calibrateEssay surfaces clamping metadata and digBench85 matches strictly-below >= 85% percentile', () => {
  const normalEssay = E.calibrateEssay(5.5, 1.5);
  assert.equal(normalEssay.clampedSd, null);
  assert.ok(Math.abs(normalEssay.effectiveMean - 5.5) < 0.01);
  assert.ok(Math.abs(normalEssay.effectiveSd - 1.5) < 0.01);

  // Extreme combination (mean = 8.0, SD = 3.5) cannot be realized on [0, 10] and must report clampedSd === 'max'
  const clampedEssay = E.calibrateEssay(8.0, 3.5);
  assert.equal(clampedEssay.clampedSd, 'max');
  assert.ok(clampedEssay.effectiveSd < 3.48);

  // Strictly-below convention for digBench85: Personal Score Tester must report >= 85% score below
  for (const presetKey of ['baseline', 'ceilingCrunch', 'harshFatigue']) {
    const sim = E.runSimulation({ ...E.PRESETS.baseline, ...E.PRESETS[presetKey] });
    const evAtBench = sim.evaluatePersonalScore(15, 20, sim.digBench85, 7.0);
    assert.ok(
      evAtBench.percentiles.digPart2 >= 0.85 - 1e-12,
      `Preset ${presetKey}: digBench85=${sim.digBench85} had strictly-below pct=${evAtBench.percentiles.digPart2} < 0.85`
    );
    if (sim.digBench85 > 0) {
      const evBelow = sim.evaluatePersonalScore(15, 20, sim.digBench85 - 1, 7.0);
      assert.ok(
        evBelow.percentiles.digPart2 < 0.85,
        `Preset ${presetKey}: digBench85-1=${sim.digBench85 - 1} already reached ${evBelow.percentiles.digPart2} >= 0.85`
      );
    }
  }
});

test('4x4 Part 2 sitter Cholesky decouples rhoVN from PSD projection and preserves baseline Part 1 pass rate & RL sum', () => {
  // Non-PSD Part 2 correlation combination must trigger matrixAdjusted without altering Part 1 exact results
  const nonPsdPart2 = {
    ...E.PRESETS.baseline,
    rhoVP2: 0.75,
    rhoEukDig: 0.0,
    rhoEukEssay: 0.0,
    rhoDigEssay: 0.0
  };
  const simBase = E.runSimulation(E.PRESETS.baseline);
  const simNonPsd = E.runSimulation(nonPsdPart2);

  assert.equal(simBase.matrixAdjusted, false);
  assert.equal(simNonPsd.matrixAdjusted, true);
  // Part 1 exact results must be completely identical regardless of Part 2 PSD projection
  assert.equal(simNonPsd.part1.pJointPass, simBase.part1.pJointPass);
  assert.equal(simNonPsd.part1.obsCorrVN, simBase.part1.obsCorrVN);
  assert.equal(simNonPsd.estPart1Passers, simBase.estPart1Passers);

  // Baseline Part 1 pass rate ~42.4% and sum(estInReserveList) close to 1,490
  assert.ok(
    Math.abs(simBase.part1.pJointPass - 0.424) < 0.01,
    `Baseline Part 1 pass rate drifted: ${(simBase.part1.pJointPass * 100).toFixed(2)}%`
  );
  const rlSum = simBase.verbalBreakdown.reduce((acc, r) => acc + r.estInReserveList, 0);
  assert.ok(
    Math.abs(rlSum - E.CONSTANTS.N_RL) < 75,
    `Baseline sum(estInReserveList)=${rlSum.toFixed(1)} drifted from 1,490`
  );
});


