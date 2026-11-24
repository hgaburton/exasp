(* ::Package:: *)

(* ============================================================
   Single-row contour figure: Final state fidelity | Final photon number,
   both computed from ONE dataset (exact/fidelity_wmax_lmax.csv).
   The two panels touch (shared internal axis); each gets its own
   legend since fidelity and Pn have different value ranges.
   No column titles above the panels.
   ============================================================ *)

SetDirectory[NotebookDirectory[]];

(* --- Import dataset --- *)
importCSV[file_] := Module[{raw, dat, l, w, f, p},
   raw = Import[file, "CSV"];
   dat = Rest[raw]; (* drop header row: lmax,wmax,fidelity,Pn *)
   l = ToExpression /@ dat[[All, 1]];
   w = ToExpression /@ dat[[All, 2]];
   f = ToExpression /@ dat[[All, 3]];
   p = ToExpression /@ dat[[All, 4]];
   <|"lmax" -> l, "wmax" -> w, "fid" -> Transpose[{l, w, f}], "pn" -> Transpose[{l, w, p}]|>
   ];

dataFile = "exact/fidelity_wmax_lmax.csv";
If[! FileExistsQ[dataFile], Print["Missing: ", dataFile]];
data = importCSV[dataFile];

(* --- Optimal (max-fidelity) point, marked on both panels --- *)
bestPoint[fidTriples_] := Module[{best}, best = First[SortBy[fidTriples, -Last[#] &]]; best[[1 ;; 2]]];
optPoint = bestPoint[data["fid"]];
Print["Optimal fidelity point (lmax,wmax): ", optPoint];

(* --- Diverging color scale (same palette as the 2x2 figure) --- *)
diverging = Blend[{RGBColor[0.85, 0.55, 0.15], RGBColor[0.55, 0.2, 0.55],
    RGBColor[0.15, 0.35, 0.75]}, #] &;

(* Theoretical bounds, not auto-computed from the (chopped) visible data: fidelity
   is physically bounded in [0,1], and Pn in [0,2] for this two-excitation subspace.
   Using the real bounds means round tick values (1, 0.8, ..., 0) land exactly at
   their true position on the bar -- no extrapolation, no misalignment between the
   two tick columns. *)
fidRange = {0, 1};
pnRange = {0, 2};

fidColorFn = diverging[Rescale[#, fidRange]] &;
pnColorFn = diverging[Rescale[#, pnRange]] &;

(* --- Panel sizing --- *)
panelWidth = 400;
plotAreaHeight = 210;
topLabelSpace = 6;     (* small buffer above frame; no title row, no top labels *)
bottomLabelSpace = 55; (* room for outer bottom tick numbers + axis label *)
plotAspectRatio = plotAreaHeight/panelWidth;

(* --- Tick helpers: major (labeled, long) + minor (unlabeled, short), inward only --- *)
ticksWithMinor[majorVals_, minorVals_, fmt_, showLabels_: True] := Join[
   ({#, If[showLabels, fmt[#], ""], {0.014, 0}} &) /@ majorVals,
   ({#, "", {0.007, 0}} &) /@ minorVals
   ];
blankTicks[ticks_] := ({#[[1]], "", If[Length[#] >= 3, #[[3]], Automatic]} &) /@ ticks;

(* y (wmax) ticks -- axis chopped at 1.2, major every 0.2 *)
yTickVals = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2};
yMinorVals = {0.1, 0.3, 0.5, 0.7, 0.9, 1.1};
yTicksReal = ticksWithMinor[yTickVals, yMinorVals, NumberForm[#, {2, 1}] &, True];

(* x (lmax) ticks -- axis chopped at 0.35, major every 0.05 (including 0.0), minor at midpoints *)
xTickVals = {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35};
xMinorVals = {0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325};
xTicksReal = ticksWithMinor[xTickVals, xMinorVals, NumberForm[#, {2, 2}] &, True];
(* Left panel's rightmost tick (0.35) sits right at the seam with the right panel
   and its number gets clipped/overlapped there -- keep the tick mark but blank the label. *)
suppressLabel[ticks_, val_] := ticks /. {v_, _, len_} /; v == val :> {v, "", len};
xTicksLeftPanel = suppressLabel[xTicksReal, 0.35];
(* Right panel's leftmost tick (0.00) sits at that same seam from its own side --
   same clipping problem, mirrored. Keep the tick mark, blank the label. *)
xTicksRightPanel = suppressLabel[xTicksReal, 0.00];

commonOpts = {
   ContourStyle -> Directive[GrayLevel[0.15], Thickness[0.0015]],
   Contours -> 14,
   ColorFunctionScaling -> False,
   InterpolationOrder -> 2,
   PlotRange -> {{0, 0.35}, {0, 1.2}},
   PlotRangePadding -> 0,
   PlotRangeClipping -> True,
   AspectRatio -> plotAspectRatio,
   Frame -> True,
   FrameStyle -> Directive[Black, Thickness[0.0025]],
   FrameTicksStyle -> Directive[13],
   LabelStyle -> Directive[13, Black]
   };

(* --- Panel builder ---
   isLeftCol controls which edges show real labels vs blank mirrored ticks.
   Both panels are "bottom row" (there's only one row), so both get the x label. *)
sidePad = 55; (* L/R padding that makes room for the y-axis label on the outer edge *)

markerFor[optPoint_] := {
   {Black, PointSize[0.017], Point[optPoint]}
   };

buildPanel[dat_, colorFn_, isLeftCol_, insetLabel_] :=
  ListContourPlot[dat,
   Evaluate[commonOpts],
   ColorFunction -> colorFn,
   ImageSize -> panelWidth,
   ImagePadding -> {
     {If[isLeftCol, sidePad, 0], If[isLeftCol, 0, sidePad]}, (* equal L+R sum for touching panels *)
     {bottomLabelSpace, topLabelSpace}
     },
   FrameLabel -> {
     {If[isLeftCol, Style["\!\(\*SubscriptBox[\(\[Omega]\), \(max\)]\)", 16, Black], None], None},
     {Style["\!\(\*SubscriptBox[\(\[Lambda]\), \(max\)]\)", 16, Black], None}
     },
   FrameTicks -> {
     {If[isLeftCol, yTicksReal, blankTicks[yTicksReal]], blankTicks[yTicksReal]},
     {If[isLeftCol, xTicksLeftPanel, xTicksRightPanel], blankTicks[xTicksReal]}
     },
   Epilog -> Join[markerFor[optPoint],
     {Inset[Style[insetLabel, 15, Black], Scaled[{0.03, 0.94}], {Left, Top}]}]
   ];

fidPlot = buildPanel[data["fid"], fidColorFn, True, "Final state fidelity"];
pnPlot = buildPanel[data["pn"], pnColorFn, False, "Final photon number"];

(* --- Legend: ONE rectangle, since both panels share the same underlying
   diverging palette applied to a 0-1 normalized fraction. Now that fidRange/pnRange
   are the clean theoretical bounds [0,1] and [0,2], round-number ticks land exactly
   at their true position -- both tick columns line up on the same rows automatically. --- *)
fidTicks = {0, 0.2, 0.4, 0.6, 0.8, 1.0};
pnTicks = {0, 0.4, 0.8, 1.2, 1.6, 2.0};

singleLegendBar[diverging_, fidRange_, pnRange_, fidTicks_, pnTicks_] := Module[
   {barW = 25, barH = plotAreaHeight, nSeg = 300, rects, fidTickPos, pnTickPos,
    fidLines, pnLines, fidLabels, pnLabels, frame, eps},
   eps = barH/nSeg*0.6; (* slight overlap between segments to kill sub-pixel gap lines *)
   rects = Table[
     {EdgeForm[None], diverging[(i - 0.5)/nSeg],
      Rectangle[{0, (i - 1)*barH/nSeg - eps}, {barW, i*barH/nSeg + eps}]},
     {i, 1, nSeg}];
   fidTickPos = ({#, Rescale[#, fidRange, {0, barH}]} &) /@ fidTicks;
   pnTickPos = ({#, Rescale[#, pnRange, {0, barH}]} &) /@ pnTicks;
   fidLines = ({Black, Thickness[0.0018], Line[{{-6, #[[2]]}, {0, #[[2]]}}]} &) /@ fidTickPos;
   pnLines = ({Black, Thickness[0.0018], Line[{{barW, #[[2]]}, {barW + 6, #[[2]]}}]} &) /@ pnTickPos;
   fidLabels = ({Text[Style[NumberForm[N[#[[1]]], {2, 1}], 14, Black], {-10, #[[2]]}, {1, 0}]} &) /@ fidTickPos;
   pnLabels = ({Text[Style[NumberForm[N[#[[1]]], {2, 1}], 14, Black], {barW + 10, #[[2]]}, {-1, 0}]} &) /@ pnTickPos;
   frame = {FaceForm[None], EdgeForm[{Black, Thickness[0.0025]}], Rectangle[{0, 0}, {barW, barH}]};
   Graphics[Join[rects, {frame}, fidLines, pnLines, fidLabels, pnLabels],
     PlotRange -> {{-45, barW + 45}, {-5, barH + 5}}, PlotRangeClipping -> False,
     ImagePadding -> {{0, 0}, {14, 4}}, ImageSize -> {95, barH}]
   ];

legendBar = singleLegendBar[diverging, fidRange, pnRange, fidTicks, pnTicks];

(* Small top spacer so the legend bar aligns with the plot area (not the padded box) *)
legendUpShift = 20;
legend = Grid[{{Pane["", {95, Max[topLabelSpace - legendUpShift, 0]}]}, {legendBar}}, Spacings -> {0, 0}];

(* --- Assemble: two touching panels, gap, then the single legend --- *)
combined = Grid[{{fidPlot, pnPlot, legend}},
   Spacings -> {{0, 0, 0.3}, 0}, Alignment -> Top]

(* --- Export --- *)
Export["exact/fidelity_pn_contour_single_test.pdf", combined, ImageSize -> 1400];
Export["exact/fidelity_pn_contour_single_test.png", combined,
  ImageSize -> 1600, ImageResolution -> 300, Background -> White];

