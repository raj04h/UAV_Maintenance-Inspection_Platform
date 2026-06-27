package com.hr.druav.ui.network

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.ai.client.generativeai.GenerativeModel
import com.google.firebase.database.FirebaseDatabase
import com.hr.druav.BuildConfig
import com.hr.druav.R
import com.hr.druav.databinding.FragmentNetworkBinding
import kotlinx.coroutines.launch

class NetworkFragment : Fragment() {

    private var _binding: FragmentNetworkBinding? = null
    private val binding get() = _binding!!

    // List to store generated error names
    private val errorList = mutableListOf<String>()
    private lateinit var adapter: ArrayAdapter<String>

    // Predefined error sets (6 for each type)
    private val gpsErrors = listOf(
        "GPS signal lost",
        "GPS module failure",
        "GPS antenna disconnected",
        "GPS data error",
        "GPS hardware malfunction",
        "Connections Successful"
    )

    private val powerErrors = listOf(
        "Battery low",
        "Power supply failure",
        "Overheating detected",
        "Power surge error",
        "Power module malfunction",
        "Fully Charged"
    )

    private val softwareErrors = listOf(
        "Firmware update failed",
        "Software crash",
        "App not responding",
        "Configuration error",
        "Software version mismatch",
        "Tested OK"
    )

    // Generative model for Gemenai API integration
    private lateinit var generativeModel: GenerativeModel

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        _binding = FragmentNetworkBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        // Initialize the Gemenai generative model similar to ScannerFragment
        generativeModel = GenerativeModel(
            modelName = "gemini-1.5-flash",
            apiKey = BuildConfig.GEMINI_API
        )

        // Setup ListView adapter for errorList
        adapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1, errorList)
        binding.listTechnicalErrors.adapter = adapter

        // Button click listeners to generate errors based on type
        binding.btnGps.setOnClickListener {
            generateRandomError("gps")
        }
        binding.btnPower.setOnClickListener {
            generateRandomError("power")
        }
        binding.btnFirm.setOnClickListener {
            generateRandomError("software")
        }

        // List item click listener to generate the solution using Gemenai for the selected error
        binding.listTechnicalErrors.setOnItemClickListener { _, _, position, _ ->
            val selectedError = errorList[position]
            binding.txtSolution.text = "Generating solution..."
            // Launch coroutine to generate solution asynchronously
            viewLifecycleOwner.lifecycleScope.launch {
                val solution = getSolutionFromGemenai(selectedError)
                binding.txtSolution.text = solution
            }
        }

        // Fetch Technician button saves the list of errors to Firebase Realtime Database
        binding.btnSave.setOnClickListener {
            saveErrorsToFirebase()
        }
    }

    // Function to generate a random error based on type and update the list
    private fun generateRandomError(type: String) {
        val randomError = when (type) {
            "gps" -> gpsErrors.random()
            "power" -> powerErrors.random()
            "software" -> softwareErrors.random()
            else -> "Unknown error"
        }
        errorList.add(randomError)
        adapter.notifyDataSetChanged()
    }

    // Function to generate a solution using the Gemenai API for a given error
    private suspend fun getSolutionFromGemenai(error: String): String {
        return try {
            val prompt = "Provide a solution for technical error: $error"
            val response = generativeModel.generateContent(prompt)
            response.text ?: "No solution found."
        } catch (e: Exception) {
            "Error retrieving solution: ${e.message}"
        }
    }

    // Function to save the error list to Firebase Realtime Database
    private fun saveErrorsToFirebase() {
        if (!isAdded || activity == null || context == null) {
            return
        }

        val database = FirebaseDatabase.getInstance()
        val errorsRef = database.getReference("errors")

        // Prepare the error data with a timestamp
        val errorData = hashMapOf<String, Any>(
            "timestamp" to System.currentTimeMillis(),
            "errors" to errorList.toList()
        )

        errorsRef.push().setValue(errorData).addOnCompleteListener { task ->
            if (!isAdded || activity == null) return@addOnCompleteListener // Ensure fragment is still attached

            if (task.isSuccessful) {
                context?.let {
                    Toast.makeText(it, "Errors saved successfully", Toast.LENGTH_SHORT).show()
                }
            } else {
                context?.let {
                    Toast.makeText(it, "Failed to save errors", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
