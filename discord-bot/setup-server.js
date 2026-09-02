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

// Vorbe's brand purple (public/index.html's --brand: #a13dff) - used as the
// accent color on every embed below, so bot messages read as genuinely
// Vorbe-branded rather than generic bot output.
const BRAND_COLOR = 0xa13dff;

// Read-only channels (below) deny @everyone the ability to post - for
// announcements/rules/welcome, where only admins should write.
//
// `key` is a stable identifier independent of the emoji-prefixed display
// name - used for cross-channel mentions (see `mention()` in main()) so
// e.g. the welcome embed can link to #help by its real channel ID rather
// than typing the literal text "#help", which Discord does NOT turn into
// a clickable mention on its own (confirmed live - it rendered as plain
// text, not a link).
//
// `embed`, where present, is a function (mention) => {title, description}
// posted by the bot once - only when the channel is genuinely empty
// (checked via lastMessageId), so re-running this script never
// double-posts or overwrites something you've since edited by hand.
const STRUCTURE = [
  ["📌 INFORMATION", [
    {
      key: "welcome", name: "👋│welcome", readOnly: true,
      topic: "Start here - what Vorbe is and how this server works.",
      embed: (mention) => ({
        title: "Welcome to the Vorbe server",
        description: [
          "Vorbe is a free, open-source AI coding assistant for Vortex developers - describe what you want, get working Luau code. Not affiliated with Vortex itself, just a community tool built for it.",
          "",
          "**Get started:** https://vorber.vercel.app  |  **Source:** https://github.com/abutauskas/Vorbe",
          "",
          `Questions go in ${mention("help")}. Found a bug? ${mention("bug-reports")} (or open a GitHub issue). Built something with Vorbe? Show it off in ${mention("showcase")}.`,
        ].join("\n"),
      }),
    },
    {
      key: "announcements", name: "📢│announcements", readOnly: true,
      topic: "Releases and project updates.",
      embed: () => ({
        title: "Server's live",
        description: "This is where Vorbe releases and project updates get posted - everything else has its own channel, check the categories on the left.",
      }),
    },
    {
      key: "rules", name: "📜│rules", readOnly: true,
      topic: "Server rules.",
      embed: () => ({
        title: "Rules",
        description: [
          "1. Be respectful - disagree without being a jerk.",
          "2. Keep it on-topic for the channel you're in.",
          "3. No spam, self-promo dumps, or unrelated advertising.",
          "4. No harassment, hate speech, or NSFW content.",
          "5. Follow Discord's own Terms of Service and Community Guidelines.",
          "",
          "Breaking these gets you a warning, then a kick or ban depending on severity.",
        ].join("\n"),
      }),
    },
  ]],
  ["👥 COMMUNITY", [
    { key: "general", name: "💬│general", topic: "General chat." },
    { key: "showcase", name: "🎨│showcase", topic: "Share what you built with Vorbe." },
    { key: "off-topic", name: "🎲│off-topic", topic: "Anything not Vorbe/Vortex related." },
  ]],
  ["🛠️ SUPPORT", [
    { key: "help", name: "🆘│help", topic: "Ask questions about using Vorbe." },
    { key: "bug-reports", name: "🐛│bug-reports", topic: "Found a bug? Also welcome as a GitHub issue: github.com/abutauskas/Vorbe/issues" },
    { key: "feature-requests", name: "💡│feature-requests", topic: "Ideas for what Vorbe should do next." },
  ]],
  ["💻 DEVELOPMENT", [
    { key: "contributing", name: "🤝│contributing", topic: "Want to help build Vorbe? Start here - see CONTRIBUTING.md." },
    { key: "dev-chat", name: "👨‍💻│dev-chat", topic: "Technical discussion for contributors." },
  ]],
];

// `colors`, not the deprecated singular `color` - confirmed against this
// project's installed discord.js source (RoleManager.js) directly.
//
// Listed highest-privilege first since Discord tends to place newly created
// roles just below the bot's own role in that order - but role hierarchy is
// ultimately a drag-to-reorder thing in Discord's own UI, so double check
// Server Settings -> Roles looks right after running this.
const ROLES = [
  {
    // No permissions on the role itself, deliberately - it's assigned only
    // to the real Discord server owner (guild.ownerId, below), who already
    // has full control regardless of any role. This is a visual tag, not a
    // grant of power, which also sidesteps a real restriction: the bot
    // can't grant a role Administrator unless the bot itself has
    // Administrator, and there's no reason to give the bot that.
    name: "Owner",
    colors: { primaryColor: 0xf1c40f }, // gold
    hoist: true,
  },
  {
    // Unlike Owner, this DOES need real permissions - it's meant for other
    // people who aren't the server owner. The bot needs these same
    // permissions itself to grant them (Discord won't let it hand out a
    // permission it doesn't hold), which is why the bot's invite gets
    // expanded below rather than just creating this role with the bot's
    // current (narrower) permission set.
    name: "Moderator",
    colors: { primaryColor: 0x3498db }, // blue
    hoist: true,
    permissions: [
      PermissionsBitField.Flags.KickMembers,
      PermissionsBitField.Flags.BanMembers,
      PermissionsBitField.Flags.ManageMessages,
      PermissionsBitField.Flags.ModerateMembers, // timeout
      PermissionsBitField.Flags.ManageNicknames,
      PermissionsBitField.Flags.ViewAuditLog,
    ],
  },
  { name: "Contributor", colors: { primaryColor: 0x9b59b6 }, hoist: true },  // purple, shown separately in the member list
  { name: "Verified", colors: { primaryColor: 0x2ecc71 }, hoist: false },    // green
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

  // Assign Owner to whoever Discord itself considers the real owner
  // (guild.ownerId) - not a hardcoded/asked-for user ID, so this is always
  // correct even if the script is handed to someone else's server later.
  const ownerRole = guild.roles.cache.find((r) => r.name === "Owner");
  if (ownerRole) {
    const ownerMember = await guild.members.fetch(guild.ownerId);
    if (ownerMember.roles.cache.has(ownerRole.id)) {
      console.log(`role  "Owner" already assigned to ${ownerMember.user.tag}`);
    } else {
      await ownerMember.roles.add(ownerRole);
      console.log(`role  "Owner" assigned to ${ownerMember.user.tag}`);
    }
  }

  console.log("");

  // Pass 1: create everything, fix permissions, and record each channel by
  // its stable `key` - needed before any embed is built, since mentions
  // need real channel IDs that only exist once every channel in STRUCTURE
  // has actually been created (a channel referenced early in the list can
  // still need to mention one defined later).
  const channelsByKey = {};

  for (const [categoryName, entries] of STRUCTURE) {
    let category = guild.channels.cache.find(
      (c) => c.type === ChannelType.GuildCategory && c.name === categoryName
    );
    if (category) {
      console.log(`category "${categoryName}" already exists`);
    } else {
      category = await guild.channels.create({ name: categoryName, type: ChannelType.GuildCategory });
      console.log(`category "${categoryName}" created`);
    }

    for (const entry of entries) {
      let channel = guild.channels.cache.find(
        (c) => c.type === ChannelType.GuildText && c.name === entry.name && c.parentId === category.id
      );

      if (channel) {
        console.log(`  #${entry.name} already exists`);
      } else {
        // The @everyone deny below also blocks the bot itself (it only has
        // Manage Channels/Manage Roles, not Administrator) - the explicit
        // allow for the bot's own ID is what lets it post the embed below
        // into a channel nobody else can write to.
        const permissionOverwrites = entry.readOnly
          ? [
              { id: guild.roles.everyone.id, deny: [PermissionsBitField.Flags.SendMessages] },
              { id: client.user.id, allow: [PermissionsBitField.Flags.SendMessages] },
            ]
          : [];

        channel = await guild.channels.create({
          name: entry.name,
          type: ChannelType.GuildText,
          parent: category.id,
          topic: entry.topic,
          permissionOverwrites,
        });
        console.log(`  #${entry.name} created${entry.readOnly ? " (read-only)" : ""}`);
      }

      // Applied every run, not just at creation - fixes channels that
      // already existed before this bot-allow overwrite was added (exactly
      // what happened on this project's own server: the channels were
      // created first, this fix came after).
      if (entry.readOnly) {
        const botOverwrite = channel.permissionOverwrites.cache.get(client.user.id);
        if (!botOverwrite || !botOverwrite.allow.has(PermissionsBitField.Flags.SendMessages)) {
          await channel.permissionOverwrites.edit(client.user.id, { SendMessages: true });
          console.log(`    fixed bot's own send-permission in #${entry.name}`);
        }
      }

      channelsByKey[entry.key] = channel;
    }
  }

  // Pass 2: post embeds now that every channel (and therefore every real
  // ID a mention might need) exists. `mention(key)` returns a real,
  // clickable <#id> mention when the target channel exists, or falls back
  // to plain "#key" text (still readable, just not a link) if it doesn't -
  // e.g. if STRUCTURE is ever edited to reference a key that was removed.
  console.log("");
  const mention = (key) => (channelsByKey[key] ? `<#${channelsByKey[key].id}>` : `#${key}`);

  for (const [, entries] of STRUCTURE) {
    for (const entry of entries) {
      if (!entry.embed) continue;
      const channel = channelsByKey[entry.key];

      // Only seed if the channel is genuinely empty - never overwrites or
      // duplicates something already posted, by the bot or by a human.
      // Deliberately NOT using channel.lastMessageId here: Discord doesn't
      // roll that field back when the last message is deleted, so it stays
      // non-null forever once anything's ever been posted (confirmed live -
      // this bit the exact case of "delete the old message, expect it to
      // re-seed" while building this). An actual fetch is the only
      // reliable check.
      const recent = await channel.messages.fetch({ limit: 1 });
      if (recent.size > 0) continue;

      const built = entry.embed(mention);
      await channel.send({ embeds: [{ color: BRAND_COLOR, ...built }] });
      console.log(`embed posted in #${entry.name}`);
    }
  }

  console.log("\nDone.");
  process.exit(0);
}

main().catch((err) => {
  console.error("Setup failed:", err);
  process.exit(1);
});
