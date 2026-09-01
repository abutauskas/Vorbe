// Vorbe Discord server setup - one-time (re-runnable) provisioning script.
// Creates the categories, channels, and roles for the community server.
// Safe to run more than once: skips anything that already exists by name,
// so it won't create duplicates if you re-run it after tweaking STRUCTURE
// below.
//
// Requires BOT_TOKEN and GUILD_ID in .env - see .env.example and
// ../scripts/setup-discord-server.sh for the guided walkthrough.

require("dotenv").config();
const { Client, GatewayIntentBits, PermissionsBitField, ChannelType } = require("discord.js");

const BOT_TOKEN = process.env.BOT_TOKEN;
const GUILD_ID = process.env.GUILD_ID;

if (!BOT_TOKEN || !GUILD_ID) {
  console.error("Missing BOT_TOKEN or GUILD_ID - set them in discord-bot/.env (see .env.example)");
  process.exit(1);
}

// Read-only channels (below) deny @everyone the ability to post - for
// announcements/rules/welcome, where only admins should write.
const STRUCTURE = [
  ["INFORMATION", [
    ["welcome", { readOnly: true, topic: "Start here - what Vorbe is and how this server works." }],
    ["announcements", { readOnly: true, topic: "Releases and project updates." }],
    ["rules", { readOnly: true, topic: "Server rules." }],
  ]],
  ["COMMUNITY", [
    ["general", { topic: "General chat." }],
    ["showcase", { topic: "Share what you built with Vorbe." }],
    ["off-topic", { topic: "Anything not Vorbe/Vortex related." }],
  ]],
  ["SUPPORT", [
    ["help", { topic: "Ask questions about using Vorbe." }],
    ["bug-reports", { topic: "Found a bug? Also welcome as a GitHub issue: github.com/abutauskas/Vorbe/issues" }],
    ["feature-requests", { topic: "Ideas for what Vorbe should do next." }],
  ]],
  ["DEVELOPMENT", [
    ["contributing", { topic: "Want to help build Vorbe? Start here - see CONTRIBUTING.md." }],
    ["dev-chat", { topic: "Technical discussion for contributors." }],
  ]],
];

const ROLES = [
  { name: "Contributor", color: 0x9b59b6, hoist: true },  // purple, shown separately in the member list
  { name: "Verified", color: 0x2ecc71, hoist: false },    // green
];

async function main() {
  const client = new Client({ intents: [GatewayIntentBits.Guilds] });
  await client.login(BOT_TOKEN);
  console.log(`Logged in as ${client.user.tag}`);

  const guild = await client.guilds.fetch(GUILD_ID);
  await guild.channels.fetch();
  await guild.roles.fetch();
  console.log(`Provisioning "${guild.name}"...\n`);

  // Roles first, in case a future run wants to reference them in channel
  // permission overwrites.
  for (const roleDef of ROLES) {
    const existing = guild.roles.cache.find((r) => r.name === roleDef.name);
    if (existing) {
      console.log(`role  "${roleDef.name}" already exists, skipping`);
      continue;
    }
    await guild.roles.create(roleDef);
    console.log(`role  "${roleDef.name}" created`);
  }

  console.log("");

  for (const [categoryName, channels] of STRUCTURE) {
    let category = guild.channels.cache.find(
      (c) => c.type === ChannelType.GuildCategory && c.name === categoryName
    );
    if (category) {
      console.log(`category "${categoryName}" already exists`);
    } else {
      category = await guild.channels.create({ name: categoryName, type: ChannelType.GuildCategory });
      console.log(`category "${categoryName}" created`);
    }

    for (const [channelName, opts] of channels) {
      const existing = guild.channels.cache.find(
        (c) => c.type === ChannelType.GuildText && c.name === channelName && c.parentId === category.id
      );
      if (existing) {
        console.log(`  #${channelName} already exists, skipping`);
        continue;
      }

      const permissionOverwrites = opts.readOnly
        ? [{ id: guild.roles.everyone.id, deny: [PermissionsBitField.Flags.SendMessages] }]
        : [];

      await guild.channels.create({
        name: channelName,
        type: ChannelType.GuildText,
        parent: category.id,
        topic: opts.topic,
        permissionOverwrites,
      });
      console.log(`  #${channelName} created${opts.readOnly ? " (read-only)" : ""}`);
    }
  }

  console.log("\nDone.");
  process.exit(0);
}

main().catch((err) => {
  console.error("Setup failed:", err);
  process.exit(1);
});
