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
  markedEukMean: number;
  markedDigMean: number;
  markedEssayMean: number;
  markedEssayPassRate: number;
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
  calibrateEssay: (targetMean: number, targetSd: number) => { muStar: number; sdStar: number };
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

test('Baseline simulation produces realistic ~42.5% Part 1 pass rate, attenuated r_obs, exact PSD matrix, and monotonic RL odds', () => {
  assert.equal(E.PRESETS.baseline.attrition, 15);
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
  assert.equal(E.PART2_KEYS.length, 11);

  const baseSim = E.runSimulation(E.PRESETS.baseline);
  const vEasierSim = E.runSimulation(E.PRESETS.jppVerbalEasier);
  const naEasierSim = E.runSimulation(E.PRESETS.jppNaEasier);
  const lowPassSim = E.runSimulation(E.PRESETS.lowPassRate);
  const p2BaseSim = E.runSimulation(E.PRESETS.part2Baseline);
  const ceilingSim = E.runSimulation(E.PRESETS.ceilingCrunch);
  const harshSim = E.runSimulation(E.PRESETS.harshFatigue);

  // Scopes & natural PSD
  assert.equal(E.PRESETS.jppVerbalEasier.scope, 'part1');
  assert.equal(E.PRESETS.jppNaEasier.scope, 'part1');
  assert.equal(E.PRESETS.lowPassRate.scope, 'part1');
  assert.equal(E.PRESETS.part2Baseline.scope, 'part2');
  assert.equal(E.PRESETS.ceilingCrunch.scope, 'part2');
  assert.equal(E.PRESETS.harshFatigue.scope, 'part2');

  for (const s of [baseSim, vEasierSim, naEasierSim, lowPassSim, p2BaseSim, ceilingSim, harshSim]) {
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

  // Low pass rate within 33.0% +/- 1.0%
  assert.ok(
    Math.abs(lowPassSim.part1.pJointPass - 0.33) <= 0.01,
    `Expected ~33.0% low pass rate, got ${(lowPassSim.part1.pJointPass * 100).toFixed(2)}%`
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
  assert.equal(E.PRESETS.baseline.eukMean, 19.0);
  assert.equal(E.PRESETS.baseline.digMean, 27.0);
  assert.equal(E.PRESETS.baseline.digSd, 5.5);
  assert.equal(E.PRESETS.ceilingCrunch.eukMean, 21.0);
  assert.equal(E.PRESETS.ceilingCrunch.digMean, 29.5);
  assert.equal(E.PRESETS.ceilingCrunch.digSd, 5.5);
  assert.equal(E.PRESETS.harshFatigue.name, 'Harsh Calibration');
  assert.equal(E.PRESETS.harshFatigue.digMean, 25.0);
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
});
