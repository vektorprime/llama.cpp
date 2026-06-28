<script lang="ts">
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import { LayoutTemplate, Trash2, MessageSquarePlus, X, Loader2, FolderPlus, Pencil, Check, Folder as FolderIcon, FolderOpen } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { ActionIcon } from '$lib/components/app';
	import { promptTemplatesStore } from '$lib/stores/prompt-templates.svelte';
	import { ROUTES } from '$lib/constants';
	import { browser } from '$app/environment';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import type { ApiPromptTemplateMeta } from '$lib/types/api';

	interface Props {
		class?: string;
	}

	let { class: className }: Props = $props();

	let previousRouteId = $state<string | null>(null);
	let editingId = $state<string | null>(null);
	let editingName = $state('');
	let expandedFolders = $state<Set<string>>(new Set());

	$effect(() => {
		const currentId = page.route.id;
		return () => { previousRouteId = currentId; };
	});

	function handleClose() {
		const prevIsTemplates = previousRouteId === '/prompt-templates';
		if (browser && window.history.length > 1 && !prevIsTemplates) {
			history.back();
		} else {
			goto(ROUTES.START);
		}
	}

	let templates = $derived(promptTemplatesStore.templates);
	let loading = $derived(promptTemplatesStore.loading);
	let error = $derived(promptTemplatesStore.error);

	let folders = $derived.by(() => {
		const set = new Set<string>();
		for (const t of templates) {
			if (t.folder) set.add(t.folder);
		}
		return [...set].sort();
	});

	let groupedTemplates = $derived.by(() => {
		const uncategorized: ApiPromptTemplateMeta[] = [];
		const folderMap = new Map<string, ApiPromptTemplateMeta[]>();
		for (const t of templates) {
			if (t.folder) {
				const list = folderMap.get(t.folder) || [];
				list.push(t);
				folderMap.set(t.folder, list);
			} else {
				uncategorized.push(t);
			}
		}
		return { uncategorized, folderMap };
	});

	function handleStartChat(template: ApiPromptTemplateMeta) {
		goto(`/?template_id=${encodeURIComponent(template.id)}#/`);
	}

	function startRename(template: ApiPromptTemplateMeta) {
		editingId = template.id;
		editingName = template.name || template.id;
	}

	async function commitRename() {
		if (!editingId || !editingName.trim()) {
			editingId = null;
			return;
		}
		try {
			await promptTemplatesStore.updateTemplate(editingId, { name: editingName.trim() });
		} catch (e) {
			console.error('Failed to rename template:', e);
		}
		editingId = null;
	}

	function handleRenameKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') commitRename();
		if (e.key === 'Escape') { editingId = null; }
	}

	async function moveToFolder(template: ApiPromptTemplateMeta, folder: string | null) {
		try {
			await promptTemplatesStore.updateTemplate(template.id, { folder: folder || '' });
		} catch (e) {
			console.error('Failed to move template:', e);
		}
	}

	function toggleFolder(folder: string) {
		const next = new Set(expandedFolders);
		if (next.has(folder)) next.delete(folder);
		else next.add(folder);
		expandedFolders = next;
	}

	async function handleDelete(template: ApiPromptTemplateMeta) {
		try {
			await promptTemplatesStore.deleteTemplate(template.id);
		} catch (e: unknown) {
			console.error('Failed to delete template:', e);
		}
	}

	function formatDate(timestamp: number): string {
		return new Date(timestamp * 1000).toLocaleString();
	}

	let displayName = (template: ApiPromptTemplateMeta) => template.name || template.id;

	onMount(() => {
		promptTemplatesStore.fetchTemplates();
	});
</script>

<div in:fade={{ duration: 150 }}>
	<div class="fixed top-4.5 right-4 z-50 md:hidden">
		<ActionIcon icon={X} tooltip="Close" onclick={handleClose} />
	</div>

	<div class="sticky top-0 z-10 mt-4 mb-2 flex items-start gap-4 md:p-4 p-0 px-4 md:justify-between md:px-8">
		<div class="flex items-center gap-2">
			<LayoutTemplate class="h-5 w-5 md:h-6 md:w-6" />
			<h1 class="text-lg font-semibold md:text-2xl">Prompt Templates</h1>
		</div>
	</div>

	<div class="grid gap-5 md:space-y-4 {className}">
		{#if loading}
			<div class="flex items-center justify-center py-12">
				<Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
			</div>
		{:else if error}
			<div class="rounded-md border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
				{error}
			</div>
		{:else if templates.length === 0}
			<div class="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
				<LayoutTemplate class="mx-auto mb-3 h-8 w-8 opacity-40" />
				<p class="font-medium">No prompt templates saved yet</p>
				<p class="mt-1 text-xs">Use the + button in chat to save the current conversation as a reusable template.</p>
			</div>
		{:else}
			<!-- Uncategorized templates -->
			{#if groupedTemplates.uncategorized.length > 0}
				<div class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(min(360px, calc(100dvw - 2rem)), 1fr));">
					{#each groupedTemplates.uncategorized as template (template.id)}
						{@render templateCard(template)}
					{/each}
				</div>
			{/if}

			<!-- Folder groupings -->
			{#each folders as folder}
				{@const items = groupedTemplates.folderMap.get(folder) || []}
				{#if items.length > 0}
					<div class="space-y-2">
						<button
							type="button"
							class="flex items-center gap-2 rounded-md px-2 py-1 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
							onclick={() => toggleFolder(folder)}
						>
							{#if expandedFolders.has(folder)}
								<FolderOpen class="h-4 w-4" />
							{:else}
								<Folder class="h-4 w-4" />
							{/if}
							{folder}
							<span class="text-xs text-muted-foreground">({items.length})</span>
						</button>
						{#if expandedFolders.has(folder)}
							<div class="grid gap-3 pl-2" style="grid-template-columns: repeat(auto-fill, minmax(min(360px, calc(100dvw - 2rem)), 1fr));">
								{#each items as template (template.id)}
									{@render templateCard(template)}
								{/each}
							</div>
						{/if}
					</div>
				{/if}
			{/each}
		{/if}
	</div>
</div>

{#snippet templateCard(template: ApiPromptTemplateMeta)}
	<Card class="flex flex-col">
		<CardHeader class="pb-2">
			<div class="flex items-start justify-between gap-2">
				<div class="min-w-0 flex-1">
					{#if editingId === template.id}
						<Input
							class="h-7 text-sm font-medium"
							bind:value={editingName}
							onkeydown={handleRenameKeydown}
							onblur={commitRename}
							autofocus
						/>
					{:else}
						<CardTitle
							class="text-sm font-medium truncate cursor-pointer hover:text-primary transition-colors"
							title="Click to rename"
							onclick={() => startRename(template)}
							ondblclick={() => startRename(template)}
						>
							{displayName(template)}
						</CardTitle>
					{/if}
				</div>
				<div class="flex items-center gap-1 shrink-0">
					<Button
						variant="ghost"
						size="icon"
						class="h-7 w-7 text-muted-foreground hover:text-primary"
						onclick={() => startRename(template)}
						aria-label="Rename template"
					>
						<Pencil class="h-3.5 w-3.5" />
					</Button>
					<Button
						variant="ghost"
						size="icon"
						class="h-7 w-7 text-muted-foreground hover:text-destructive"
						onclick={() => handleDelete(template)}
						aria-label="Delete template"
					>
						<Trash2 class="h-3.5 w-3.5" />
					</Button>
				</div>
			</div>
			<CardDescription class="text-xs">
				{formatDate(template.created_at)}
			</CardDescription>
		</CardHeader>
		<CardContent class="pt-0 pb-3">
			<div class="flex flex-wrap gap-1.5 mb-3">
				<Badge variant="secondary" class="text-xs">
					{template.token_count} tokens
				</Badge>
				<Badge variant="outline" class="text-xs">
					{template.model_name}
				</Badge>
				<Badge variant="outline" class="text-xs">
					K:{template.cache_type_k} V:{template.cache_type_v}
				</Badge>
				{#if template.n_swa > 0 && !template.swa_full}
					<Badge variant="destructive" class="text-xs">SWA limited</Badge>
				{/if}
			</div>
			<div class="text-xs text-muted-foreground space-y-0.5 mb-2">
				<div>Context: {template.context_size} | Vocab type: {template.model_vocab_type}</div>
				{#if template.folder}
					<div class="flex items-center gap-1">
						<Folder class="h-3 w-3" /> {template.folder}
					</div>
				{/if}
			</div>
			<div class="mt-3 flex gap-2">
				<Button
					variant="default"
					size="sm"
					class="flex-1"
					onclick={() => handleStartChat(template)}
				>
					<MessageSquarePlus class="mr-1.5 h-4 w-4" />
					Start Chat
				</Button>
			</div>
		</CardContent>
	</Card>
{/snippet}
