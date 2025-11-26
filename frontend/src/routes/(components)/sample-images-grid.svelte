<script lang="ts">
  import { cn } from "$lib/utils";
  import { API_BASE_URL } from "$lib/api";
  import type { ProcessedSample, Sample } from "$lib/types";

  interface Props {
    maxSamples: number;
    samples: Sample[];
    processedSamples: ProcessedSample[] | undefined;
  }

  let { maxSamples, samples = $bindable(), processedSamples }: Props = $props();
</script>

<div class="grid gap-6 mt-6 grid-cols-4">
  {#each samples.slice(0, maxSamples) as sample, index}
    {@const is_correct = processedSamples?.[index].prediction.is_correct}

    <div class={cn("flex relative items-center rounded-lg overflow-hidden justify-center aspect-square transition ring-transparent ring-4", is_correct === true && "ring-green-500", is_correct === false && "ring-rose-600")}>
      <div class="absolute inset-0 bg-accent animate-pulse"></div>
      <img class="size-full z-0 object-cover saturate-115" src="{API_BASE_URL}{sample.url}" alt="" />
      <div class="absolute top-2.5 left-2.5 z-10 text-xs bg-white/80 backdrop-blur-3xl text-foreground px-1.5 py-0.5 rounded">GT:{sample.class_name}</div>
    </div>
  {/each}
</div>
