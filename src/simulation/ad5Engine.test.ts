import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

interface EnginePreset {
  name: string;
  vMean: number;
  vSd: number;
  naMean: number;
  naSd: number;
  rhoVN: number;
  attrition: number;
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
  baselinePasserOdds: number;
  matrixAdjusted: boolean;
  prelimCutoff: number;
  finalCutoff: number;
  markedEssayMean: number;
  greenThresholdV: number;
  part1: {
    pVerbalPass: number;
    pNAPass: number;
    pJointPass: number;
    obsCorrVN: number;
    verbalAllPmf: Float64Array;
    verbalPassPmf: Float64Array;
  };
  verbalBreakdown: Array<{
    v: number;
    estPassers: number;
    pNAGivenV: number;
    condRlOdds: number;
    overallOddsPreNA: number;
    reqEuk: number;
    prelimGateBinds: boolean;
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
  };
}

interface AD5EngineApi {
  CONSTANTS: {
    REGISTERED: number;
    N_SIT: number;
    N_RL: number;
    N_PRELIM: number;
  };
  PRESETS: Record<string, EnginePreset>;
  calcPrelimScore: (v: number, euk: number, dig: number) => number;
  calcFinalScore: (v: number, euk: number, dig: number, essay: number) => number;
  calibrate3PL: (key: string, mean: number, sd: number, passMark: number) => Calibrated3PL;
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

test('Baseline simulation produces realistic ~42.5% Part 1 pass rate, attenuated r_obs, and monotonic RL odds', () => {
  const sim = E.runSimulation(E.PRESETS.baseline);

  // Part 1 pass rate near 42.5%
  assert.ok(
    sim.part1.pJointPass > 0.39 && sim.part1.pJointPass < 0.46,
    `Unexpected baseline joint pass rate: ${sim.part1.pJointPass}`
  );
  // Observed correlation strictly attenuated below latent rhoVN = 0.60
  assert.ok(
    sim.part1.obsCorrVN > 0.35 && sim.part1.obsCorrVN < 0.54,
    `Unexpected observed correlation: ${sim.part1.obsCorrVN}`
  );
  // Marked Top 2,235 essay mean must be strictly higher than pool mean (6.2) due to positive correlation
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

test('Bidirectional Joint Pass Rate solver shifts both means equally and clamps gracefully at extremes', () => {
  const base = E.PRESETS.baseline;
  const solved35 = E.solveMeansForPassRate(base, 0.35);
  assert.ok(Math.abs(solved35.achievedPassRate - 0.35) < 0.005);
  // Equal shift preserves (vMean - naMean) = 0.10
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
