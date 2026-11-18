# Publication Figures for IEEE Research Paper

This directory contains all high-resolution (300 DPI) figures formatted for insertion into `IEEE_RESEARCH_PAPER_FROM_DARKNESS_TO_DETAIL.tex`.

## File Index & Mapping

| File Name | LaTeX Figure Reference | Description | Recommended Width |
|:---|:---|:---|:---|
| `fig1_architecture_pipeline.png` | `\ref{fig:arch_diagram}` | Overall system schematic of Proposed Encoder–Decoder with PixelShuffle | `\linewidth` (single col) or `1.5\linewidth` |
| `fig2_qualitative_comparisons.png` | `\ref{fig:visual_comparisons}` | 5-column qualitative comparison grid (Input, NAFNet, NAFNet+CA, Proposed, GT) | `\linewidth` (double col `figure*`) |
| `fig3_pareto_efficiency.png` | `\ref{fig:pareto}` | Pareto efficiency scatter plot (Parameters vs. GPU Latency vs. PSNR) | `\linewidth` (single col) |
| `fig4_training_loss.png` | Fig in Sec IV-A | Training composite loss minimization trajectories over 150 epochs | `\linewidth` |
| `fig5_validation_loss.png` | Fig in Sec IV-A | Validation loss convergence on the 50-pair validation split | `\linewidth` |
| `fig6_validation_psnr.png` | Fig in Sec IV-B | Validation PSNR trajectories illustrating peak checkpoint convergence | `\linewidth` |
| `fig7_validation_ssim.png` | Fig in Sec IV-B | Validation SSIM structural fidelity trajectories | `\linewidth` |
| `fig8_architecture_comparison_bars.png` | Fig in Sec IV-B | Validation vs. Official Test (`eval15`) grouped bar chart comparison | `\linewidth` |
| `fig9_per_image_test_distribution.png` | Fig in Sec IV-B | Per-image PSNR and SSIM distribution on official `eval15` test set | `\linewidth` |
| `fig10_visual_grid_encoder_decoder.png` | Appendix Visuals | Visual comparison grid for Proposed Encoder–Decoder | `\linewidth` |
| `fig11_visual_grid_nafnet_channel.png` | Appendix Visuals | Visual comparison grid for NAFNet + Channel Attention | `\linewidth` |
| `fig12_visual_grid_nafnet_baseline.png` | Appendix Visuals | Visual comparison grid for NAFNet Baseline | `\linewidth` |
| `fig13_visual_grid_nafnet_gating.png` | Appendix Visuals | Visual comparison grid for NAFNet + Adaptive Gating | `\linewidth` |
| `fig14_loss_curves_encoder_decoder.png` | Appendix Curves | Loss and validation trajectory curves for Encoder–Decoder | `\linewidth` |
| `fig15_loss_curves_nafnet_channel.png` | Appendix Curves | Loss and validation trajectory curves for NAFNet + Channel Attention | `\linewidth` |

---
### How to Link in LaTeX:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures_for_paper/fig1_architecture_pipeline.png}
  \caption{System block diagram of the proposed Encoder--Decoder network.}
  \label{fig:arch_diagram}
\end{figure}
```
