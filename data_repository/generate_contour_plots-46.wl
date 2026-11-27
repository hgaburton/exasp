(* ::Package:: *)

(* ============================================================
   2x2 contour figure: Artificial vs Physical (columns) x
   Final photon number vs Final state fidelity (rows).
   ============================================================ *)

SetDirectory[NotebookDirectory[]];

(* --- Import both datasets --- *)
importCSV[file_] := Module[{raw, hdr, dat, l, w, f, p},
   raw = Import[file, "CSV"];
   dat = Rest[raw];
   l = ToExpression /@ dat[[All, 1]];
   w = ToExpression /@ dat[[All, 2]];
   f = ToExpression /@ dat[[All, 3]];
   p = ToExpression /@ dat[[All, 4]];
   <|"lmax" -> l, "wmax" -> w, "fid" -> Transpose[{l, w, f}], "pn" -> Transpose[{l, w, p}]|>
   ];

artFile = "numerical_values/fidelity_wmax_lmax_extended_more.csv";
physFile = "numerical_values/fidelity_wmax_lmax_physical.csv";
If[! FileExistsQ[artFile], Print["Missing: ", artFile]];
If[! FileExistsQ[physFile], Print["Missing: ", physFile]];

art = importCSV[artFile];
phys = importCSV[physFile];

(* --- Optimal (max-fidelity) point per column --- *)
bestPoint[fidTriples_] := Module[{best}, best = First[SortBy[fidTriples, -Last[#] &]]; best[[1 ;; 2]]];
(* Hardcoded manually -- set these to whatever {lmax, wmax} you want the dot at: *)
optArt = {2.0, 5.3};   (* {lmax, wmax} for the Artificial panel *)
optPhys = {1.4, 5.3};  (* {lmax, wmax} for the Physical panel *)
Print["Artificial optimal fidelity point (lmax,wmax): ", optArt];
Print["Physical optimal fidelity point (lmax,wmax): ", optPhys];

(* --- Extend all four datasets so optimal-point dots near the right edge sit inside real plotted colour --- *)
lmaxDomainMax = Max[Join[art["lmax"], phys["lmax"]]];
extendDelta = 0.04*(lmaxDomainMax - Min[Join[art["lmax"], phys["lmax"]]]);

extendRight[triples_] := Module[{edgeRows, extraRows},
   edgeRows = Select[triples, Abs[#[[1]] - lmaxDomainMax] < 10^-6 &];
   extraRows = ({lmaxDomainMax + extendDelta, #[[2]], #[[3]]} &) /@ edgeRows;
   Join[triples, extraRows]
   ];

art["pn"] = extendRight[art["pn"]]; art["fid"] = extendRight[art["fid"]];
phys["pn"] = extendRight[phys["pn"]]; phys["fid"] = extendRight[phys["fid"]];

(* Operating point *)
operatingPoint = {0.3, 5.8};

(* --- Diverging color scale --- *)
diverging = Blend[{RGBColor[0.85, 0.55, 0.15], RGBColor[0.55, 0.2, 0.55],
    RGBColor[0.15, 0.35, 0.75]}, #] &;

(* --- Shared color ranges PER ROW (so both columns in a row use the SAME
   value-to-color mapping, since they share one legend) --- *)
pnRange = {Min[Join[art["pn"][[All, 3]], phys["pn"][[All, 3]]]],
   Max[Join[art["pn"][[All, 3]], phys["pn"][[All, 3]]]]};
fidRange = {Min[Join[art["fid"][[All, 3]], phys["fid"][[All, 3]]]],
   Max[Join[art["fid"][[All, 3]], phys["fid"][[All, 3]]]]};

(* ColorFunctionScaling->False + manual Rescale ensures identical color<->value
   mapping across both columns of a row (independent per-plot auto-scaling would
   silently make the same color mean different things in each column). *)
pnColorFn = diverging[Rescale[#, pnRange]] &;
fidColorFn = diverging[Rescale[#, fidRange]] &;

(* --- Panel sizing --- *)
panelWidth = 400;
plotAreaHeight = 210;
topLabelSpace = 6;    (* small buffer above frame; top edge has no numeric labels *)
bottomLabelSpace = 55; (* room for outer bottom tick numbers + axis label *)
plotAspectRatio = plotAreaHeight/panelWidth;

topRowHeight = plotAreaHeight + topLabelSpace;
bottomRowHeight = plotAreaHeight + bottomLabelSpace;

(* --- Tick helpers: major (labeled, long) + minor (unlabeled, short) --- *)
ticksWithMinor[majorVals_, minorVals_, fmt_, showLabels_: True] := Join[
   ({#, If[showLabels, fmt[#], ""], {0.014, 0}} &) /@ majorVals,
   ({#, "", {0.007, 0}} &) /@ minorVals
   ];
blankTicks[ticks_] := ({#[[1]], "", If[Length[#] >= 3, #[[3]], Automatic]} &) /@ ticks;

(* y (wmax) ticks: shared between rows (both datasets assumed same wmax domain) *)
wmaxMin = Min[Join[art["wmax"], phys["wmax"]]];
wmaxMax = Max[Join[art["wmax"], phys["wmax"]]];
yStep = 2;
allYVals = Range[Ceiling[wmaxMin, yStep], Floor[wmaxMax, yStep], yStep];
yMinorVals = Range[Ceiling[wmaxMin, yStep] + yStep/2, Floor[wmaxMax, yStep] - yStep/2, yStep];
topYTicks = ticksWithMinor[allYVals, yMinorVals, NumberForm[#, {2, 0}] &, True];
bottomYTicks = ticksWithMinor[Select[allYVals, # < wmaxMax - 10^-6 &], yMinorVals, NumberForm[#, {2, 0}] &, True];

(* x (lmax) ticks: explicit values, shown only on bottom row *)
xTickVals = {0.0, 0.4, 0.8, 1.2, 1.6, 2.0};
xMinorVals = {0.2, 0.6, 1.0, 1.4, 1.8};
xTicksLabeled = ticksWithMinor[xTickVals, xMinorVals, NumberForm[#, {2, 1}] &, True];
xTicksBlank = ticksWithMinor[xTickVals, xMinorVals, NumberForm[#, {2, 1}] &, False];

commonOpts = {
   ContourStyle -> Directive[GrayLevel[0.15], Thickness[0.0015]],
   Contours -> 14,
   ColorFunctionScaling -> False,
   InterpolationOrder -> 2,
   PlotRangePadding -> 0,
   PlotRangeClipping -> False,
   AspectRatio -> plotAspectRatio,
   Frame -> True,
   FrameStyle -> Directive[Black, Thickness[0.0025]],
   FrameTicksStyle -> Directive[13],
   LabelStyle -> Directive[13, Black]
   };

markerFor[optPoint_] := {
   {Black, PointSize[0.017], Point[optPoint]}
   };

(* --- Panel builder ---
   isLeftCol / isBottomRow control which edges show real labels vs blank mirrored ticks.
   titleText -> None suppresses the inset row title (used on right column). *)
sidePad = 55; (* L/R padding that makes room for the y-axis label on the outer edge *)

buildPanel[data_, colorFn_, optPoint_, isLeftCol_, isBottomRow_, titleText_] :=
  ListContourPlot[data,
   Evaluate[commonOpts],
   ColorFunction -> colorFn,
   ImageSize -> panelWidth,
   ImagePadding -> {
     {If[isLeftCol, sidePad, 0], If[isLeftCol, 0, sidePad]}, (* equal L+R sum (sidePad) for both columns:
        ImageSize is a single width number, so AspectRatio derives height from
        (panelWidth - leftPad - rightPad); unequal sums silently gave different
        heights per column, which is what broke vertical touching last time *)
     {If[isBottomRow, bottomLabelSpace, 0], If[isBottomRow, 0, topLabelSpace]}
     },
   FrameLabel -> {
     {If[isLeftCol, Style["\!\(\*SubscriptBox[\(\[Omega]\), \(max\)]\)", 16, Black], None], None},
     {If[isBottomRow, Style["\!\(\*SubscriptBox[\(\[Lambda]\), \(max\)]\)", 16, Black], None], None}
     },
   FrameTicks -> {
     {If[isLeftCol, If[isBottomRow, bottomYTicks, topYTicks], blankTicks[If[isBottomRow, bottomYTicks, topYTicks]]],
      blankTicks[If[isBottomRow, bottomYTicks, topYTicks]]},
     {If[isBottomRow, xTicksLabeled, xTicksBlank], xTicksBlank}
     },
   Epilog -> Join[markerFor[optPoint],
     If[titleText === None, {},
       {Inset[Style[titleText, 15, Black], Scaled[{0.03, 0.94}], {Left, Top}]}]]
   ];

(* --- Build all four panels --- *)
pnPhysPlot = buildPanel[phys["pn"], pnColorFn, optPhys, True, False, "Final photon number"];
pnArtPlot = buildPanel[art["pn"], pnColorFn, optArt, False, False, None];
fidPhysPlot = buildPanel[phys["fid"], fidColorFn, optPhys, True, True, "Final state fidelity"];
fidArtPlot = buildPanel[art["fid"], fidColorFn, optArt, False, True, None];

(* --- Legends: one per row, height matched to that row's total panel height --- *)
pnTicks = ({#, NumberForm[#, {2, 1}]} &) /@ {0, 0.4, 0.8, 1.2, 1.6, 2.0};
fidTicks = ({#, NumberForm[#, {2, 1}]} &) /@ {0, 0.2, 0.4, 0.6, 0.8, 1.0};

(* --- Legends: bar height matches the actual PLOT AREA (not the padded box), with a
   blank spacer that mirrors each panel's own asymmetric padding *)
pnLegendBar = BarLegend[{diverging, pnRange},
   LegendLabel -> None, Ticks -> pnTicks,
   LegendMarkerSize -> {25, plotAreaHeight},
   LabelStyle -> Directive[14, Black], ImageSize -> {65, plotAreaHeight}];
fidLegendBar = BarLegend[{diverging, fidRange},
   LegendLabel -> None, Ticks -> fidTicks,
   LegendMarkerSize -> {25, plotAreaHeight},
   LabelStyle -> Directive[14, Black], ImageSize -> {65, plotAreaHeight}];

legendUpShift = 20; (* how far to pull the whole legend block up *)
pnLegend = Grid[{{Pane["", {65, Max[topLabelSpace - legendUpShift, 0]}]}, {pnLegendBar}}, Spacings -> {0, 0}];
fidLegend = Grid[{{fidLegendBar}, {Pane["", {65, bottomLabelSpace}]}}, Spacings -> {0, 0}];

(* --- Column title headers --- *)
colTitle[text_, xOffset_: 0] := Graphics[Text[Style[text, 16, Black], {xOffset, 0}],
   PlotRange -> {{-2.5, 2.5}, {-0.5, 0.5}}, AspectRatio -> Full,
   ImageSize -> {panelWidth, 20}, ImagePadding -> {{10, 10}, {2, 2}},
   PlotRangeClipping -> False];
titleOffsetUnits = (sidePad/2)*(5/panelWidth);

(* --- Isolate the plot panels into their own Grid, separate from the legends. --- *)
panelGrid = Grid[{{pnPhysPlot, pnArtPlot}, {fidPhysPlot, fidArtPlot}},
   Spacings -> {0, 0}, Alignment -> Left];
legendGrid = Grid[{{pnLegend}, {fidLegend}}, Spacings -> {0, 0}, Alignment -> Top];
titleGrid = Grid[{{colTitle["Physical", titleOffsetUnits], colTitle["Artificial", -titleOffsetUnits]}}, Spacings -> {0}];

combined = Grid[{
    {titleGrid, ""},
    {panelGrid, legendGrid}
    },
   Spacings -> {0.4, 0}, Alignment -> {Left, Top}]

(* --- Export --- *)
Export["numerical_values/fidelity_pn_contours_2x2.pdf", combined, ImageSize -> 1400];
Export["numerical_values/fidelity_pn_contours_2x2.png", combined,
  ImageSize -> 2000, ImageResolution -> 300, Background -> White];

