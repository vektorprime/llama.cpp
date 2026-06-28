<script lang="ts">
	import { DialogModelNotAvailable } from '$lib/components/app';
import { chatStore } from '$lib/stores/chat.svelte';
import { conversationsStore, isConversationsInitialized } from '$lib/stores/conversations.svelte';
import { modelsStore, modelOptions } from '$lib/stores/models.svelte';
import { promptTemplatesStore } from '$lib/stores/prompt-templates.svelte';
import { DatabaseService } from '$lib/services';
import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';
	import { APP_NAME, NEW_CHAT_PARAM } from '$lib/constants';

	let qParam = $derived(page.url.searchParams.get('q'));
	let modelParam = $derived(page.url.searchParams.get('model'));
	let newChatParam = $derived(page.url.searchParams.get(NEW_CHAT_PARAM));
	let templateIdParam = $derived(page.url.searchParams.get('template_id'));

	// Dialog state for model not available error
	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelOptions().map((m) => m.model));

	/**
	 * Clear URL params after message is sent to prevent re-sending on refresh
	 */
	function clearUrlParams() {
		const url = new URL(page.url);

		url.searchParams.delete('q');
		url.searchParams.delete('model');
		url.searchParams.delete(NEW_CHAT_PARAM);
		url.searchParams.delete('template_id');

		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		await modelsStore.fetch();

		if (modelParam) {
			const model = modelsStore.findModelByName(modelParam);

			if (model) {
				try {
					await modelsStore.selectModelById(model.id);
				} catch (error) {
					console.error('Failed to select model:', error);
					requestedModelName = modelParam;
					showModelNotAvailable = true;

					return;
				}
			} else {
				requestedModelName = modelParam;
				showModelNotAvailable = true;

				return;
			}
		}

		// Handle ?q= parameter - create new conversation and send message
		if (qParam !== null) {
			await conversationsStore.createConversation();
			clearUrlParams();
		} else if (modelParam || newChatParam === 'true' || templateIdParam) {
			if (templateIdParam) {
				let restoredCount = 0;
				try {
					const template = await promptTemplatesStore.getTemplate(templateIdParam);
					if (template.messages && template.messages.length > 0) {
						await conversationsStore.createConversation();
						const conv = conversationsStore.activeConversation;
						if (conv) {
							const rootId = await DatabaseService.createRootMessage(conv.id);
							let parentId = rootId;
							for (const msg of template.messages) {
								if (msg.role === 'system') {
									const dbMsg = await DatabaseService.createSystemMessage(conv.id, msg.content, parentId);
									conversationsStore.addMessageToActive(dbMsg);
									parentId = dbMsg.id;
								} else if (msg.role === 'user') {
									const dbMsg = await DatabaseService.createMessageBranch({
										convId: conv.id,
										type: 'text' as const,
										role: msg.role,
										content: msg.content,
										timestamp: Date.now(),
										toolCalls: '',
										children: [],
									}, parentId);
									conversationsStore.addMessageToActive(dbMsg);
									parentId = dbMsg.id;
								} else if (msg.role === 'assistant') {
									const dbMsg = await DatabaseService.createMessageBranch({
										convId: conv.id,
										type: 'text' as const,
										role: msg.role,
										content: msg.content,
										timestamp: Date.now(),
										toolCalls: '',
										children: [],
									}, parentId);
									conversationsStore.addMessageToActive(dbMsg);
									parentId = dbMsg.id;
								}
								restoredCount++;
							}
							await conversationsStore.updateCurrentNode(parentId);
						}
					}
				} catch (e) {
					console.error('Failed to restore template messages:', e);
				}
				chatStore.setPendingTemplate(templateIdParam, restoredCount);
			}
			clearUrlParams();
		}
	}

	onMount(async () => {
		if (!isConversationsInitialized()) {
			await conversationsStore.initialize();
		}

		conversationsStore.clearActiveConversation();
		chatStore.clearUIState();

		await modelsStore.fetch();

		if (qParam !== null || modelParam !== null || newChatParam === 'true' || templateIdParam !== null) {
			await handleUrlParams();
		}

		await modelsStore.ensureFirstModelSelected();
	});
</script>

<svelte:head>
	<title>{APP_NAME}</title>
</svelte:head>

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	modelName={requestedModelName}
	availableModels={availableModelNames}
/>
