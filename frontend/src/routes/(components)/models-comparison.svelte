<script lang="ts">
  import { cn } from "$lib/utils";
  import Skeleton from "$lib/components/ui/skeleton/skeleton.svelte";
  import type { Model, ProcessingResult } from "$lib/types";

  interface Props {
    processingResults: ProcessingResult[];
    selectedProcessingResult: ProcessingResult | undefined;
    isProcessingResults: boolean;
    selectedModels: Model[];
  }

  let { processingResults, selectedProcessingResult = $bindable(), isProcessingResults, selectedModels }: Props = $props();

  function floatToPercentageString(float: number) {
    return `${(float * 100).toFixed(2)}%`;
  }

  let scores = $derived(processingResults.map((r) => r.metrics.accuracy + r.metrics.f1_score));
  let maxScore = $derived(scores.length ? Math.max(...scores) : -1);
  let bestIndices = $derived(scores.map((s, i) => (s === maxScore ? i : null)).filter((i) => i !== null));
</script>

<div class="mt-12">
  <h2 class="font-bold">Performance Metrics</h2>
  <p class="text-sm mt-0.5 text-muted-foreground">Model metrics and comparison for the current sample set</p>
</div>

<div class="grid rounded-lg mb-3 border mt-6 grid-cols-[auto_auto_auto_auto_auto_auto_auto_auto_auto_auto] overflow-hidden">
  {@render headerCell("")}
  {@render headerCell("Model")}
  {@render headerCell("Accuracy")}
  {@render headerCell("Precision")}
  {@render headerCell("Recall")}
  {@render headerCell("F1-Score")}
  {@render headerCell("TP")}
  {@render headerCell("TN")}
  {@render headerCell("FP")}
  {@render headerCell("FN")}

  {#if isProcessingResults}
    {#each { length: selectedModels.length } as _}
      <div class="px-3 py-3 flex items-center justify-center">
        <div class="size-4 rounded-full shrink-0 bg-muted shadow-xs aspect-square"></div>
      </div>

      <div class="px-3 py-3">
        <Skeleton class="w-24 h-4 my-0.5" />
      </div>

      {#each { length: 8 } as _}
        <div class="px-3 py-3">
          <Skeleton class="w-full h-4 my-0.5" />
        </div>
      {/each}
    {/each}
  {:else}
    {#each processingResults as processingResult, index}
      {@render radio(index)}
      {@render cell(processingResult.model.name + (bestIndices.includes(index) ? " 🏆" : ""), index, "font-medium")}
      {@render cell(floatToPercentageString(processingResult.metrics.accuracy), index)}
      {@render cell(floatToPercentageString(processingResult.metrics.precision), index)}
      {@render cell(floatToPercentageString(processingResult.metrics.recall), index)}
      {@render cell(floatToPercentageString(processingResult.metrics.f1_score), index)}
      {@render cell(processingResult.metrics.tp, index)}
      {@render cell(processingResult.metrics.tn, index)}
      {@render cell(processingResult.metrics.fp, index)}
      {@render cell(processingResult.metrics.fn, index)}
    {/each}
  {/if}
</div>

{#snippet radio(index: number)}
  {@const processingResultModelName = processingResults[index]?.model.name}
  {@const selectedProcessingResultModelName = selectedProcessingResult?.model.name}

  <div class="flex items-center justify-end">
    <button onclick={() => (selectedProcessingResult = processingResults[index])} title="radio" class="border-input flex items-center justify-center text-primary shadow-xs aspect-square size-4 shrink-0 rounded-full border transition">
      <div class={cn("relative rounded-full size-2 opacity-0 transition bg-primary", selectedProcessingResultModelName && selectedProcessingResultModelName === processingResultModelName && "opacity-100")}></div>
    </button>
  </div>
{/snippet}

{#snippet cell(text: string | number, index: number, className?: string)}
  <button class={cn("text-sm text-left px-3 py-3", className)} onclick={() => (selectedProcessingResult = processingResults[index])}>{text}</button>
{/snippet}

{#snippet headerCell(text: string | number, className?: string)}
  <div class={cn("text-xs font-medium tracking-wide text-muted-foreground bg-muted uppercase px-3 py-3", className)}>{text}</div>
{/snippet}
