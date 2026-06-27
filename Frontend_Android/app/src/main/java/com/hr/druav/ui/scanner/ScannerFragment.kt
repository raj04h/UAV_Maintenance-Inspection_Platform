package com.hr.druav.ui.scanner

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.ai.client.generativeai.GenerativeModel
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.firebase.database.FirebaseDatabase
import com.hr.druav.BuildConfig
import com.hr.druav.databinding.FragmentScannerBinding
import com.hr.druav.ml.UavDamage
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.Executors

class ScannerFragment : Fragment() {

    private val geminai = BuildConfig.GEMINI_API
    private var _binding: FragmentScannerBinding? = null
    private val binding get() = _binding!!
    private val CAMERA_REQUEST_CODE = 100
    private val QR_REQUEST_CODE = 200

    // Change executor to a var so we can reinitialize if needed
    private var executor = Executors.newSingleThreadExecutor()
    private lateinit var generativeModel: GenerativeModel
    private val detectedProblems = mutableListOf<String>()
    private lateinit var problemAdapter: ArrayAdapter<String>

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentScannerBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        generativeModel = GenerativeModel(
            modelName = "gemini-1.5-flash",
            apiKey = geminai
        )

        problemAdapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1, detectedProblems)
        binding.listProblemDetails.adapter = problemAdapter

        binding.btnQr.setOnClickListener { openCameraForDamage() }
        binding.btnRfid.setOnClickListener { openCameraForQr() }
        binding.btnSave.setOnClickListener {
            if (detectedProblems.isNotEmpty()) {
                getSolutionForDamage(detectedProblems.last())
                saveScanToFirebase("damage", detectedProblems.last())
            } else {
                binding.TVResult.text = "No damage detected yet."
            }
        }
        binding.listProblemDetails.setOnItemClickListener { _, _, position, _ ->
            getSolutionForDamage(detectedProblems[position])
        }
    }

    private fun openCameraForDamage() {
        if (ContextCompat.checkSelfPermission(requireContext(), android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE
            )
        } else {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, CAMERA_REQUEST_CODE)
        }
    }

    private fun openCameraForQr() {
        if (ContextCompat.checkSelfPermission(requireContext(), android.Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(android.Manifest.permission.CAMERA),
                QR_REQUEST_CODE
            )
        } else {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, QR_REQUEST_CODE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                CAMERA_REQUEST_CODE -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    binding.imgHighlighted.setImageBitmap(imageBitmap)
                    analyzeImage(imageBitmap)
                }
                QR_REQUEST_CODE -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    scanQrCode(imageBitmap)
                }
            }
        }
    }

    private fun getSolutionForDamage(damageType: String) {
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val response = generativeModel.generateContent("Provide a solution for UAV damage type: $damageType")
                val solution = response.text ?: "No solution found."

                // Check if the fragment's view is still available
                if (_binding != null) {
                    activity?.runOnUiThread {
                        binding.TVResult.text = solution
                    }
                }
            } catch (e: Exception) {
                if (_binding != null) {
                    binding.TVResult.text = "Error retrieving solution: ${e.message}"
                }
            }
        }
    }


    private fun analyzeImage(bitmap: Bitmap) {
        // Reinitialize executor if it was shutdown.
        if (executor.isShutdown) {
            executor = Executors.newSingleThreadExecutor()
        }
        executor.execute {
            try {
                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(scaledBitmap)
                tensorImage.buffer.rewind()

                val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                inputFeature.loadBuffer(tensorImage.buffer)

                val model = UavDamage.newInstance(requireContext())
                val outputs = model.process(inputFeature)
                val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray
                model.close()

                val damageCategories = listOf("Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches", "No Damage")
                val maxIndex = outputArray.indices.maxByOrNull { outputArray[it] } ?: 0
                val damageType = damageCategories.getOrNull(maxIndex) ?: "Unknown"
                detectedProblems.add(damageType)

                activity?.runOnUiThread {
                    problemAdapter.notifyDataSetChanged()
                }
            } catch (e: Exception) {
                Log.e("UAV Damage Detection", "Error processing image: ${e.localizedMessage}")
            }
        }
    }

    private fun scanQrCode(bitmap: Bitmap) {
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            val options = BarcodeScannerOptions.Builder()
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                .build()
            val scanner = BarcodeScanning.getClient(options)
            scanner.process(image)
                .addOnSuccessListener { barcodes ->
                    if (barcodes.isNotEmpty()) {
                        val qrValue = barcodes[0].rawValue ?: "No value"
                        detectedProblems.add(qrValue)
                        activity?.runOnUiThread {
                            problemAdapter.notifyDataSetChanged()
                        }
                        saveScanToFirebase("QR", qrValue)
                    } else {
                        binding.TVResult.text = "No QR code found."
                    }
                }
                .addOnFailureListener { e ->
                    binding.TVResult.text = "QR scanning failed: ${e.message}"
                }
        } catch (e: Exception) {
            binding.TVResult.text = "Error processing QR image: ${e.message}"
        }
    }

    // Save scan results to Firebase.
    // Save scan results to Firebase with a toast notification.
    private fun saveScanToFirebase(type: String, value: String) {
        val database = FirebaseDatabase.getInstance("https://dr-uav-default-rtdb.europe-west1.firebasedatabase.app/")
            .getReference("Error Found")

        val scanData = mapOf(
            "type" to type,
            "value" to value,
            "timestamp" to System.currentTimeMillis()
        )

        database.push().setValue(scanData)
            .addOnSuccessListener {
                Log.d("Firebase", "Data saved successfully!")
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), "Error saved successfully!", Toast.LENGTH_SHORT).show()
                }
            }
            .addOnFailureListener { e ->
                Log.e("Firebase", "Error saving data: ${e.message}")
                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), "Failed to save data: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
    }


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        // Shutdown the executor when the view is destroyed.
        executor.shutdown()
    }
}