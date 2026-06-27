package com.hr.druav

import android.content.Intent
import android.hardware.camera2.params.BlackLevelPattern
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.ResponseStoppedException
import com.hr.druav.databinding.ActivityChatbotBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.graphics.Color
import java.util.Locale

class BotActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    private lateinit var binding: ActivityChatbotBinding
    private lateinit var textToSpeech: TextToSpeech
    private val geminai = BuildConfig.GEMINI_API
    private val VOICE_RECOGNITION_REQUEST_CODE = 1001
    private var isTtsInitialized = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate the layout for Activity
        binding = ActivityChatbotBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize Text-to-Speech
        textToSpeech = TextToSpeech(this, this)

        // Set up Send Button for text input (do not speak response)
        binding.btnPromptsend.setOnClickListener {
            if (!isTtsInitialized) {
                Toast.makeText(this, "TTS engine not ready yet!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            stopSpeaking()  // Stop any ongoing speech before processing new input
            val prompt = binding.ETPrompt.text.toString().trim()
            if (prompt.isNotEmpty()) {
                addMessage(isUser = true, message = prompt)
                binding.ETPrompt.text.clear()
                generateText(prompt, speakResponse = false)
            }
        }

        // Set up Voice Button for voice input (speak response)
        binding.btnVoice.setOnClickListener {
            if (!isTtsInitialized) {
                Toast.makeText(this, "TTS engine not ready yet!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            stopSpeaking()  // Stop ongoing speech before starting voice recognition
            startVoiceRecognition()
        }
    }

    /**
     * Start Android’s speech recognition to capture user voice input.
     */
    private fun startVoiceRecognition() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak now...")
        }
        startActivityForResult(intent, VOICE_RECOGNITION_REQUEST_CODE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == VOICE_RECOGNITION_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            val results = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
            if (!results.isNullOrEmpty()) {
                val spokenText = results[0]
                binding.ETPrompt.setText(spokenText)
                addMessage(isUser = true, message = spokenText)
                generateText(spokenText, speakResponse = true)
            }
        }
    }

    /**
     * Generate bot response using the generative model in a coroutine.
     * @param speakResponse: if true, the bot's response will be spoken using TTS.
     */
    private fun generateText(prompt: String, speakResponse: Boolean) {
        val generativeModel = GenerativeModel(
            modelName = "gemini-2.0-flash",
            apiKey = geminai
        )
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = generativeModel.generateContent(prompt)
                withContext(Dispatchers.Main) {
                    // Use a default value if response.text is null
                    val botResponse = response.text ?: "No response"
                    addMessage(isUser = false, message = botResponse)
                    if (speakResponse) {
                        stopSpeaking()
                        speakOut(botResponse)
                    }
                }
            } catch (e: ResponseStoppedException) {
                Log.e("AIError", "Content generation stopped: ${e.response}")
            } catch (e: Exception) {
                Log.e("AIError", "Error: ${e.message}")
            }
        }
    }

    /**
     * Dynamically add a chat bubble to the chat container.
     * isUser: true for user messages (aligned to right), false for bot messages (aligned to left)
     */
    private fun addMessage(isUser: Boolean, message: String) {
        val textView = TextView(this).apply {
            text = message
            setTextColor(Color.BLACK)
            textSize = 16f
            setPadding(16, 16, 16, 16)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = 8
                bottomMargin = 8
                gravity = if (isUser) Gravity.END else Gravity.START
            }
            if (isUser) {
                // Replace with your drawable resource for user messages if desired
                setBackgroundResource(R.drawable.side_nav_bar)
            } else {
                setBackgroundResource(R.drawable.bubble_background)
            }
        }

        binding.chatContainer.addView(textView)

        binding.scrollviewres.post {
            binding.scrollviewres.fullScroll(ScrollView.FOCUS_DOWN)
        }
    }

    /**
     * Initialize Text-to-Speech.
     */
    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = textToSpeech.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("TTS", "Language not supported!")
            } else {
                isTtsInitialized = true
            }
        } else {
            Log.e("TTS", "Initialization failed!")
        }
    }

    /**
     * Speak the provided text using Text-to-Speech.
     */
    private fun speakOut(text: String) {
        Log.d("TTS", "Speaking: $text")
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
    }

    /**
     * Stop any ongoing Text-to-Speech speech.
     */
    private fun stopSpeaking() {
        if (::textToSpeech.isInitialized && textToSpeech.isSpeaking) {
            textToSpeech.stop()
        }
    }

    override fun onDestroy() {
        stopSpeaking()
        textToSpeech.shutdown()
        super.onDestroy()
    }
}
