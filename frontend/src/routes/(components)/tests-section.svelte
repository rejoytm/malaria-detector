<script lang="ts">
  import { toast } from "svelte-sonner";
  import Play from "@lucide/svelte/icons/play";
  import Shuffle from "@lucide/svelte/icons/shuffle";
  import { getProcessingResults, getRandomSamples } from "$lib/api";
  import Button from "$lib/components/ui/button/button.svelte";
  import { Spinner } from "$lib/components/ui/spinner/index.js";
  import type { Dataset, Model, ProcessingResult, Sample } from "$lib/types";
  import DatasetSelector from "./dataset-selector.svelte";
  import ModelsComparison from "./models-comparison.svelte";
  import SampleImagesGrid from "./sample-images-grid.svelte";

  interface Props {
    datasets: Dataset[];
    selectedDataset: Dataset | undefined;
    selectedModels: Model[];
    samples: Sample[];
    processingResults: ProcessingResult[] | undefined;
    selectedProcessingResult: ProcessingResult | undefined;
  }

  const maxSamples = 12;

  let { datasets, selectedDataset = $bindable(), selectedModels, samples = $bindable(), processingResults, selectedProcessingResult = $bindable() }: Props = $props();

  let isProcessingResults: boolean = $state(false);

  $effect(() => {
    selectedDataset && randomizeSamples(selectedDataset);
  });

  async function randomizeSamples(dataset: Dataset) {
    const samplesResponse = await getRandomSamples(dataset.name, maxSamples);

    if (samplesResponse.data) {
      samples = samplesResponse.data;
      selectedProcessingResult = undefined;
    }
  }

  async function processSamples() {
    isProcessingResults = true;
    const processingResultsResponse = await getProcessingResults(samples, selectedModels);

    if (processingResultsResponse.error) {
      toast.error(processingResultsResponse.error.name, { description: processingResultsResponse.error.message });
    } else {
      processingResults = processingResultsResponse.data;
      selectedProcessingResult = processingResults[0];
      console.log(processingResults);
    }

    isProcessingResults = false;
  }
</script>

<div class="bg-background rounded-xl shadow py-4 px-6">
  <div class="flex justify-between gap-3 items-end">
    <div>
      <h2 class="font-bold">Test Samples</h2>
      <p class="text-sm text-muted-foreground mt-0.5">Randomize and run models on sample images</p>
    </div>

    <div class="flex gap-3">
      <DatasetSelector {datasets} bind:selectedDataset />
      <Button size="sm" disabled={!selectedDataset} variant="outline" onclick={() => selectedDataset && randomizeSamples(selectedDataset)}>
        <Shuffle />
        Randomize
      </Button>
      <Button
        class="min-w-36"
        size="sm"
        onclick={() => {
          processSamples();
        }}
      >
        {#if isProcessingResults}
          <Spinner /> Processing...
        {:else}
          <Play /> Run detection
        {/if}
      </Button>
    </div>
  </div>

  <SampleImagesGrid {maxSamples} {samples} processedSamples={selectedProcessingResult?.processed_samples} />

  {#if processingResults}
    <ModelsComparison {processingResults} bind:selectedProcessingResult {isProcessingResults} {selectedModels} />
  {/if}
</div>
